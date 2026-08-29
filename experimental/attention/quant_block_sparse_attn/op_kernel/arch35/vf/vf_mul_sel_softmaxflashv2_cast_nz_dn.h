/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn.h
 * \brief
 */

#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#include "kernel_tensor.h"
#include "vf_basic_block_utils.h"
namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false

// Empty-row lse sentinel, equivalent to common.h QBSA_EMPTY_LSE_VALUE(-FLT_MAX).
constexpr float VF_QBSA_EMPTY_LSE_VALUE = -3.4028234663852886e38F;

// Layout selectors for unpacking the four FP8 lanes produced by a B32 -> B8
// cast.  The unpacked values are accumulated into the softmax denominator so
// C2's numerator and denominator use exactly the same quantized P.
constexpr static AscendC::MicroAPI::CastTrait QBSA_CAST_B8_TO_B32_ZERO = {RegLayout::ZERO, SatMode::UNKNOWN,
                                                                          MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static AscendC::MicroAPI::CastTrait QBSA_CAST_B8_TO_B32_ONE = {RegLayout::ONE, SatMode::UNKNOWN,
                                                                         MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static AscendC::MicroAPI::CastTrait QBSA_CAST_B8_TO_B32_TWO = {RegLayout::TWO, SatMode::UNKNOWN,
                                                                         MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static AscendC::MicroAPI::CastTrait QBSA_CAST_B8_TO_B32_THREE = {RegLayout::THREE, SatMode::UNKNOWN,
                                                                           MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

// Safe pre-pad value for AttenTail variants. Must be a large negative finite float
// so that (pad_value * qScale * kScale) does NOT overflow to -inf/NaN before the
// softmax max reduction. -FLT_MAX overflows when dequant scales are large.
//
// Optimal value = -(D * FP8_MAX^2) = -(128 * 448^2) = -25,690,112.
// This is the minimum possible valid QK^T (all D dims at -448*+448), so it never
// exceeds any valid score in magnitude. After scaling by qScale*kScale*softmax_scale,
// the worst case equals -(sqrt(D) * R^2), which reaches -FP32_MAX at exactly the
// same R that causes valid scores to overflow. Thus pre-pad is never the bottleneck.
// Re-padding after the scale loop restores -FLT_MAX (vreg_min) so exp(-FLT_MAX - max) = 0.
constexpr float VF_QBSA_SAFE_PAD_VALUE = -25690112.0F;

template <typename T, typename T2, bool hasAtten = false, bool needAtten = false, uint32_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdateVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
    __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb, __ubuf__ float *qScaleUb, __ubuf__ float *kScaleUb,
    __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
    const T minValue, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, const uint32_t ubN_div_8 = 0, const uint32_t ubN_m_m_div_4 = 0,
    const uint32_t ubN_m_m_div_2 = 0, const uint32_t ubN_m_m_mul3_div_4 = 0, __ubuf__ uint32_t *qGatherIdxUb = nullptr)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_empty_lse;
    MaskReg preg_invalid_cur;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    x_exp_1 = x_exp + 32; // Cast 之后第二个寄存器搬运到UB上的目标地址需偏移32字节

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16; // 16 * 4 个元素个数(attenMaskUb 的数据类型为uint8，这里是uint32)
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32; // 32 * 4
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48; // 48 * 4

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_empty_lse, VF_QBSA_EMPTY_LSE_VALUE);
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_135);

    RegTensor<uint32_t> vreg_qgather_idx;
    MaskReg preg_qstore8;
    LoadAlign(vreg_qgather_idx, qGatherIdxUb);
    uint32_t sreg_qmask = (uint32_t)0xFFULL;
    preg_qstore8 = UpdateMask<uint16_t>(sreg_qmask);
    for (uint32_t c = 0; c < 8; ++c) {
        LoadAlign(vreg_data_tmp0, qScaleUb + c * 64); // 64 表示元素个数
        Gather(vreg_data_tmp1, vreg_data_tmp0, vreg_qgather_idx);
        StoreAlign<float>(qScaleUb + c * 8, vreg_data_tmp1, preg_qstore8); // 8 表示元素个数
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_135);
    for (uint64_t iter_m = 0; iter_m < uint64_t(ubN >> 2); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m * 4);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        if constexpr (needAtten) {
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
            Select(vreg_data_tmp0, vreg_data_tmp0, vreg_min, preg_compare0);
            Select(vreg_data_tmp1, vreg_data_tmp1, vreg_min, preg_compare1);
            Select(vreg_data_tmp2, vreg_data_tmp2, vreg_min, preg_compare2);
            Select(vreg_data_tmp3, vreg_data_tmp3, vreg_min, preg_compare3);
        }

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, vreg_data_tmp3, preg_135);
        Max(max0, max0, vreg_data_tmp0, preg_135);
        Max(max1, max1, vreg_data_tmp1, preg_135);
        Max(max2, max2, vreg_data_tmp2, preg_135);
        Max(max3, max3, vreg_data_tmp3, preg_135);
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::NE>(preg_invalid_cur, max0, minValue, preg_108);

    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Select(max0, max0, vreg_empty_lse, preg_invalid_cur);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);
    loopNum = ubN_div_8;

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m * 2);

        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64); // 64 表示元素个数，取到了vreg_x_f32_0的下一行
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m * 2 + 64);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_invalid_cur);

        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
        if constexpr (!IsSameType<T2, hifloat8_t>::value) {
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        }

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, bool needAtten = false, uint32_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdateAttenTailVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
    __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb, __ubuf__ float *qScaleUb, __ubuf__ float *kScaleUb,
    __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
    const T minValue, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, const uint32_t ubN_div_8 = 0, const uint32_t ubN_m_m_div_4 = 0,
    const uint32_t ubN_m_m_div_2 = 0, const uint32_t ubN_m_m_mul3_div_4 = 0, __ubuf__ uint32_t *qGatherIdxUb = nullptr)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_safe_min;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_empty_lse;
    MaskReg preg_invalid_cur;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    x_exp_1 = x_exp + 32; // Cast 之后第二个寄存器搬运到UB上的目标地址需偏移32字节

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16; // 16 * 4 个元素个数(attenMaskUb 的数据类型为uint8，这里是uint32)
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32; // 32 * 4
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48; // 48 * 4

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_safe_min, VF_QBSA_SAFE_PAD_VALUE);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_empty_lse, VF_QBSA_EMPTY_LSE_VALUE);
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_135);

    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * m, vreg_safe_min,
                                                          preg_135);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    RegTensor<uint32_t> vreg_qgather_idx;
    MaskReg preg_qstore8;
    LoadAlign(vreg_qgather_idx, qGatherIdxUb);
    uint32_t sreg_qmask = (uint32_t)0xFFULL;
    preg_qstore8 = UpdateMask<uint16_t>(sreg_qmask);
    for (uint32_t c = 0; c < 8; ++c) {
        LoadAlign(vreg_data_tmp0, qScaleUb + c * 64); // 64 表示元素个数
        Gather(vreg_data_tmp1, vreg_data_tmp0, vreg_qgather_idx);
        StoreAlign<float>(qScaleUb + c * 8, vreg_data_tmp1, preg_qstore8); // 8 表示元素个数
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_135);
    for (uint64_t iter_m = 0; iter_m < uint64_t(ubN >> 2); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m * 4);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        if constexpr (needAtten) {
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
            Select(vreg_data_tmp0, vreg_data_tmp0, vreg_min, preg_compare0);
            Select(vreg_data_tmp1, vreg_data_tmp1, vreg_min, preg_compare1);
            Select(vreg_data_tmp2, vreg_data_tmp2, vreg_min, preg_compare2);
            Select(vreg_data_tmp3, vreg_data_tmp3, vreg_min, preg_compare3);
        }

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, vreg_data_tmp3, preg_135);
        Max(max0, max0, vreg_data_tmp0, preg_135);
        Max(max1, max1, vreg_data_tmp1, preg_135);
        Max(max2, max2, vreg_data_tmp2, preg_135);
        Max(max3, max3, vreg_data_tmp3, preg_135);
    }

    // Tail columns may have zero scale when the second sparse block is absent.
    // Re-apply minValue after q/k scale so invalid columns cannot enter softmax.
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::NE>(preg_invalid_cur, max0, minValue, preg_108);

    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Select(max0, max0, vreg_empty_lse, preg_invalid_cur);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);
    loopNum = ubN_div_8;

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m * 2);

        LoadAlign(vreg_x_f32_4,
                  input_x_local_UB + i0 * m * 2 + 64); // 64 表示元素个数，取到了vreg_x_f32_0的下一行
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m * 2 + 64);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_invalid_cur);

        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
        if constexpr (!IsSameType<T2, hifloat8_t>::value) {
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        }

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint32_t ubN = 128>
__aicore__ inline void ProcessVec1DnNoUpdate(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &expSumTensor,
                                             const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor,
                                             const LocalTensor<T> &expMaxTensor,
                                             const LocalTensor<uint8_t> &vselrIndexesBuf,
                                             const LocalTensor<uint8_t> &maskTensor, const LocalTensor<T> &qScaleTensor,
                                             const LocalTensor<T> &kScaleTensor, const uint32_t m, const uint32_t n,
                                             const uint32_t originN, const T scale, const T minValue, bool needAtten,
                                             const float pScale = 1.0f, __ubuf__ uint32_t *qGatherIdxUb = nullptr)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ float *qScaleUb = (__ubuf__ T *)qScaleTensor.GetPhyAddr();
    __ubuf__ float *kScaleUb = (__ubuf__ T *)kScaleTensor.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    dScale = scale;
    blockStride = ubN >> 2 | 0x1; // ubN / 4 + 1，表示搬运到UB上的时候，每块会间隔32bit，避免bank冲突
    repeatStride = 2;
    indexesUb = (__ubuf__ uint8_t *)vselrIndexesBuf.GetPhyAddr();
    uint32_t ubN_div_8 = ubN >> 3;
    uint32_t ubN_m_m_div_4 = (ubN * m) >> 2;
    uint32_t ubN_m_m_div_2 = (ubN * m) >> 1;
    uint32_t ubN_m_m_mul3_div_4 = ((ubN * m) >> 1) + ((ubN * m) >> 2);

    if (needAtten && originN < ubN) {
        ProcessVec1DnNoUpdateAttenTailVF<T, T2, hasAtten, true, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale, ubN_div_8,
            ubN_m_m_div_4, ubN_m_m_div_2, ubN_m_m_mul3_div_4, qGatherIdxUb);
    } else if (originN < ubN) {
        ProcessVec1DnNoUpdateAttenTailVF<T, T2, hasAtten, false, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale, ubN_div_8,
            ubN_m_m_div_4, ubN_m_m_div_2, ubN_m_m_mul3_div_4, qGatherIdxUb);
    } else if (needAtten) {
        ProcessVec1DnNoUpdateVF<T, T2, hasAtten, true, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale, ubN_div_8,
            ubN_m_m_div_4, ubN_m_m_div_2, ubN_m_m_mul3_div_4, qGatherIdxUb);
    } else {
        ProcessVec1DnNoUpdateVF<T, T2, hasAtten, false, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale, ubN_div_8,
            ubN_m_m_div_4, ubN_m_m_div_2, ubN_m_m_mul3_div_4, qGatherIdxUb);
    }
}

template <typename T, typename T2, bool hasAtten = false, bool needAtten = false, uint32_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdateVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
    __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb, __ubuf__ float *qScaleUb, __ubuf__ float *kScaleUb,
    __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
    const T minValue, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, const uint32_t ubN_div_8 = 0, const uint32_t ubN_m_m_div_4 = 0,
    const uint32_t ubN_m_m_div_2 = 0, const uint32_t ubN_m_m_mul3_div_4 = 0, __ubuf__ uint32_t *qGatherIdxUb = nullptr)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_empty_lse;
    MaskReg preg_invalid_cur;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    x_exp_1 = x_exp + 32; // Cast 之后第二个寄存器搬运到UB上的目标地址需偏移32字节

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16; // 16 * 4 个元素个数(attenMaskUb 的数据类型为uint8，这里是uint32)
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32; // 32 * 4
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48; // 48 * 4

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_empty_lse, VF_QBSA_EMPTY_LSE_VALUE);
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_135);

    RegTensor<uint32_t> vreg_qgather_idx;
    MaskReg preg_qstore8;
    LoadAlign(vreg_qgather_idx, qGatherIdxUb);
    uint32_t sreg_qmask = (uint32_t)0xFFULL;
    preg_qstore8 = UpdateMask<uint16_t>(sreg_qmask);
    for (uint32_t c = 0; c < 8; ++c) {
        LoadAlign(vreg_data_tmp0, qScaleUb + c * 64); // 64 表示元素个数
        Gather(vreg_data_tmp1, vreg_data_tmp0, vreg_qgather_idx);
        StoreAlign<float>(qScaleUb + c * 8, vreg_data_tmp1, preg_qstore8); // 8 表示元素个数
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_135);
    for (uint64_t iter_m = 0; iter_m < uint64_t(ubN >> 2); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m * 4);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        if constexpr (needAtten) {
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
            Select(vreg_data_tmp0, vreg_data_tmp0, vreg_min, preg_compare0);
            Select(vreg_data_tmp1, vreg_data_tmp1, vreg_min, preg_compare1);
            Select(vreg_data_tmp2, vreg_data_tmp2, vreg_min, preg_compare2);
            Select(vreg_data_tmp3, vreg_data_tmp3, vreg_min, preg_compare3);
        }

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, vreg_data_tmp3, preg_135);
        Max(max0, max0, vreg_data_tmp0, preg_135);
        Max(max1, max1, vreg_data_tmp1, preg_135);
        Max(max2, max2, vreg_data_tmp2, preg_135);
        Max(max3, max3, vreg_data_tmp3, preg_135);
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::NE>(preg_invalid_cur, max0, minValue, preg_108);

    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Select(max0, max0, vreg_empty_lse, preg_invalid_cur);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);
    loopNum = ubN_div_8;

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m * 2);

        LoadAlign(vreg_x_f32_4,
                  input_x_local_UB + i0 * m * 2 + 64); // 64 表示元素个数，取到了vreg_x_f32_0的下一行
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m * 2 + 64);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_invalid_cur);

        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
        if constexpr (!IsSameType<T2, hifloat8_t>::value) {
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        }

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    RegTensor<float> vreg_l0;
    MaskReg preg_previous_sum_zero;
    LoadAlign(vreg_l0, new_global_sum);
    Compare<float, CMPMODE::EQ>(preg_previous_sum_zero, vreg_l0, vreg_zero, preg_134);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Select(vreg_l0, vreg_zero, vreg_l0, preg_previous_sum_zero);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, bool needAtten = true, uint32_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdateAttenTailVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
    __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb, __ubuf__ float *qScaleUb, __ubuf__ float *kScaleUb,
    __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
    const T minValue, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, const uint32_t ubN_div_8 = 0, const uint32_t ubN_m_m_div_4 = 0,
    const uint32_t ubN_m_m_div_2 = 0, const uint32_t ubN_m_m_mul3_div_4 = 0, __ubuf__ uint32_t *qGatherIdxUb = nullptr)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_safe_min;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_empty_lse;
    MaskReg preg_invalid_cur;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    x_exp_1 = x_exp + 32; // Cast 之后第二个寄存器搬运到UB上的目标地址需偏移32字节

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16; // 16 * 4 个元素个数(attenMaskUb 的数据类型为uint8，这里是uint32)
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32; // 32 * 4
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48; // 48 * 4

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_safe_min, VF_QBSA_SAFE_PAD_VALUE);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_empty_lse, VF_QBSA_EMPTY_LSE_VALUE);
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_135);

    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * m, vreg_safe_min,
                                                          preg_135);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    RegTensor<uint32_t> vreg_qgather_idx;
    MaskReg preg_qstore8;
    LoadAlign(vreg_qgather_idx, qGatherIdxUb);
    uint32_t sreg_qmask = (uint32_t)0xFFULL;
    preg_qstore8 = UpdateMask<uint16_t>(sreg_qmask);
    for (uint32_t c = 0; c < 8; ++c) {
        LoadAlign(vreg_data_tmp0, qScaleUb + c * 64); // 64 表示元素个数
        Gather(vreg_data_tmp1, vreg_data_tmp0, vreg_qgather_idx);
        StoreAlign<float>(qScaleUb + c * 8, vreg_data_tmp1, preg_qstore8); // 8 表示元素个数
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_135);
    for (uint64_t iter_m = 0; iter_m < uint64_t(ubN >> 2); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m * 4);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        if constexpr (needAtten) {
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
            Select(vreg_data_tmp0, vreg_data_tmp0, vreg_min, preg_compare0);
            Select(vreg_data_tmp1, vreg_data_tmp1, vreg_min, preg_compare1);
            Select(vreg_data_tmp2, vreg_data_tmp2, vreg_min, preg_compare2);
            Select(vreg_data_tmp3, vreg_data_tmp3, vreg_min, preg_compare3);
        }

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, vreg_data_tmp3, preg_135);
        Max(max0, max0, vreg_data_tmp0, preg_135);
        Max(max1, max1, vreg_data_tmp1, preg_135);
        Max(max2, max2, vreg_data_tmp2, preg_135);
        Max(max3, max3, vreg_data_tmp3, preg_135);
    }

    // Tail columns may have zero scale when the second sparse block is absent.
    // Re-apply minValue after q/k scale so invalid columns cannot enter softmax.
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::NE>(preg_invalid_cur, max0, minValue, preg_108);

    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Select(max0, max0, vreg_empty_lse, preg_invalid_cur);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);
    loopNum = ubN_div_8;

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m * 2);

        LoadAlign(vreg_x_f32_4,
                  input_x_local_UB + i0 * m * 2 + 64); // 64 表示元素个数，取到了vreg_x_f32_0的下一行
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m * 2 + 64);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_invalid_cur);

        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
        if constexpr (!IsSameType<T2, hifloat8_t>::value) {
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        }

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    RegTensor<float> vreg_l0;
    MaskReg preg_previous_sum_zero;
    LoadAlign(vreg_l0, new_global_sum);
    Compare<float, CMPMODE::EQ>(preg_previous_sum_zero, vreg_l0, vreg_zero, preg_134);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Select(vreg_l0, vreg_zero, vreg_l0, preg_previous_sum_zero);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint32_t ubN = 128>
__aicore__ inline void ProcessVec1DnUpdate(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &expSumTensor,
                                           const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor,
                                           const LocalTensor<T> &expMaxTensor,
                                           const LocalTensor<uint8_t> &vselrIndexesBuf,
                                           const LocalTensor<uint8_t> &maskTensor, const LocalTensor<T> &qScaleTensor,
                                           const LocalTensor<T> &kScaleTensor, const uint32_t m, const uint32_t n,
                                           const uint32_t originN, const T scale, const T minValue, bool needAtten,
                                           const float pScale = 1.0f, __ubuf__ uint32_t *qGatherIdxUb = nullptr)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ float *qScaleUb = (__ubuf__ T *)qScaleTensor.GetPhyAddr();
    __ubuf__ float *kScaleUb = (__ubuf__ T *)kScaleTensor.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    dScale = scale;
    blockStride = ubN >> 2 | 0x1; // ubN / 4 + 1，表示搬运到UB上的时候，每块会间隔32bit，避免bank冲突
    repeatStride = 2;
    indexesUb = (__ubuf__ uint8_t *)vselrIndexesBuf.GetPhyAddr();
    uint32_t ubN_div_8 = ubN >> 3;
    uint32_t ubN_m_m_div_4 = (ubN * m) >> 2;
    uint32_t ubN_m_m_div_2 = (ubN * m) >> 1;
    uint32_t ubN_m_m_mul3_div_4 = ((ubN * m) >> 1) + ((ubN * m) >> 2);

    if (needAtten && originN < ubN) {
        ProcessVec1DnUpdateAttenTailVF<T, T2, hasAtten, true, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale, ubN_div_8,
            ubN_m_m_div_4, ubN_m_m_div_2, ubN_m_m_mul3_div_4, qGatherIdxUb);
    } else if (originN < ubN) {
        ProcessVec1DnUpdateAttenTailVF<T, T2, hasAtten, false, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale, ubN_div_8,
            ubN_m_m_div_4, ubN_m_m_div_2, ubN_m_m_mul3_div_4, qGatherIdxUb);
    } else if (needAtten) {
        ProcessVec1DnUpdateVF<T, T2, hasAtten, true, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale, ubN_div_8,
            ubN_m_m_div_4, ubN_m_m_div_2, ubN_m_m_mul3_div_4, qGatherIdxUb);
    } else {
        ProcessVec1DnUpdateVF<T, T2, hasAtten, false, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale, ubN_div_8,
            ubN_m_m_div_4, ubN_m_m_div_2, ubN_m_m_mul3_div_4, qGatherIdxUb);
    }
}

/*
 * @ingroup ProcessVec1Vf
 * @brief 计算 max = reducemax 以及 exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor 输出 LocalTensor
 * @param [out] expSumTensor 最后一维 sum(exp(x-max)) 输出
 * @param [out] maxTensor 最后一维 max 输出
 * @param [in] srcTensor 输入 LocalTensor
 * @param [out] expMaxTensor expmax 输出 LocalTensor
 * @param [in] sharedTmpBuffer 本地临时 Tensor
 * @param [in] m 输入行数
 * @param [in] n 输入列数，需要 256B 对齐，取值为 originN 按 64 对齐后的结果
 * @param [in] originN 原始输入列数，支持范围为 0 < originN <= 128
 * @param [in] scale scale 值
 * @param [in] minValue 最小值
 * @param [in] isUpdate 是否启用 flash update 模式
 * @param [in] oriNRange originN 范围
 */

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint32_t ubN = 256>
__aicore__ inline void ProcessVec1VfDn(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &expSumTensor,
                                       const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor,
                                       const LocalTensor<T> &expMaxTensor, TBuf<> *vselrIndexesBuf,
                                       const LocalTensor<uint8_t> &maskTensor, const LocalTensor<T> &qScaleTensor,
                                       const LocalTensor<T> &kScaleTensor, const uint32_t m, const uint32_t n,
                                       const uint32_t originN, const T scale, const T minValue, bool needAtten,
                                       const float pScale = 1.0f)
{
    __ubuf__ uint32_t *qGatherIdxUb =
        (__ubuf__ uint32_t *)(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::QSCALE_GATHER_INDEX)]
                                  .template Get<uint32_t>()
                                  .GetPhyAddr());
    if constexpr (!isUpdate) {
        LocalTensor<uint8_t> indexesTensor;
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        ProcessVec1DnNoUpdate<T, T2, hasAtten, ubN>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                                                    indexesTensor, maskTensor, qScaleTensor, kScaleTensor, m, n,
                                                    originN, scale, minValue, needAtten, pScale, qGatherIdxUb);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        ProcessVec1DnUpdate<T, T2, hasAtten, ubN>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                                                  indexesTensor, maskTensor, qScaleTensor, kScaleTensor, m, n, originN,
                                                  scale, minValue, needAtten, pScale, qGatherIdxUb);
    }
}

template <typename T, typename T2, bool hasAtten = false, bool needAtten = false, uint32_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdatePerTokenHeadVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
    __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb, __ubuf__ float *qScaleUb, __ubuf__ float *kScaleUb,
    __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
    const T minValue, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f)
{
    constexpr uint32_t mConst = 64;                                                        // m = 64
    constexpr uint32_t m2Const = 128;                                                      // m * 2 = 128
    constexpr uint32_t m3Const = 192;                                                      // m * 3 = 192
    constexpr uint32_t m4Const = 256;                                                      // m * 4 = 256
    constexpr uint32_t ubN_m_m_div_4 = (ubN * mConst) >> 2;                                // ubN * m / 4
    constexpr uint32_t ubN_m_m_div_2 = (ubN * mConst) >> 1;                                // ubN * m / 2
    constexpr uint32_t ubN_m_m_mul3_div_4 = ((ubN * mConst) >> 1) + ((ubN * mConst) >> 2); // ubN * m * 3 / 4
    constexpr uint16_t loopNum = ubN >> 3;                                                 // ubN / 8

    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_empty_lse;
    MaskReg preg_invalid_cur;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    x_exp_1 = x_exp + 32; // Cast 之后第二个寄存器搬运到UB上的目标地址需偏移32字节

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + mConst;
    __ubuf__ float *src_ub2 = src_ub0 + m2Const;
    __ubuf__ float *src_ub3 = src_ub0 + m3Const;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16; // 16 * 4 个元素个数(attenMaskUb 的数据类型为uint8，这里是uint32)
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32; // 32 * 4
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48; // 48 * 4

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_empty_lse, VF_QBSA_EMPTY_LSE_VALUE);
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_135);

    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_135);
    for (uint64_t iter_m = 0; iter_m < uint64_t(ubN >> 2); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m4Const);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);
        if constexpr (needAtten) {
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * mConst);
            Select(vreg_data_tmp0, vreg_data_tmp0, vreg_min, preg_compare0);
            Select(vreg_data_tmp1, vreg_data_tmp1, vreg_min, preg_compare1);
            Select(vreg_data_tmp2, vreg_data_tmp2, vreg_min, preg_compare2);
            Select(vreg_data_tmp3, vreg_data_tmp3, vreg_min, preg_compare3);
        }

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m4Const, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m4Const, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m4Const, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m4Const, vreg_data_tmp3, preg_135);
        Max(max0, max0, vreg_data_tmp0, preg_135);
        Max(max1, max1, vreg_data_tmp1, preg_135);
        Max(max2, max2, vreg_data_tmp2, preg_135);
        Max(max3, max3, vreg_data_tmp3, preg_135);
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::NE>(preg_invalid_cur, max0, minValue, preg_108);

    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Select(max0, max0, vreg_empty_lse, preg_invalid_cur);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m2Const);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const);

        LoadAlign(vreg_x_f32_4,
                  input_x_local_UB + i0 * m2Const + mConst); // 64 表示元素个数，取到了vreg_x_f32_0的下一行
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const + mConst);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_invalid_cur);

        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
        if constexpr (!IsSameType<T2, hifloat8_t>::value) {
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        }

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, bool needAtten = false, uint32_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdatePerTokenHeadAttenTailVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
    __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb, __ubuf__ float *qScaleUb, __ubuf__ float *kScaleUb,
    __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
    const T minValue, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f)
{
    constexpr uint32_t mConst = 64;                                                        // m = 64
    constexpr uint32_t m2Const = 128;                                                      // m * 2 = 128
    constexpr uint32_t m3Const = 192;                                                      // m * 3 = 192
    constexpr uint32_t m4Const = 256;                                                      // m * 4 = 256
    constexpr uint32_t ubN_m_m_div_4 = (ubN * mConst) >> 2;                                // ubN * m / 4
    constexpr uint32_t ubN_m_m_div_2 = (ubN * mConst) >> 1;                                // ubN * m / 2
    constexpr uint32_t ubN_m_m_mul3_div_4 = ((ubN * mConst) >> 1) + ((ubN * mConst) >> 2); // ubN * m * 3 / 4
    constexpr uint16_t loopNum = ubN >> 3;                                                 // ubN / 8

    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_safe_min;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_empty_lse;
    MaskReg preg_invalid_cur;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    x_exp_1 = x_exp + 32; // Cast 之后第二个寄存器搬运到UB上的目标地址需偏移32字节

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + mConst;
    __ubuf__ float *src_ub2 = src_ub0 + m2Const;
    __ubuf__ float *src_ub3 = src_ub0 + m3Const;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16; // 16 * 4 个元素个数(attenMaskUb 的数据类型为uint8，这里是uint32)
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32; // 32 * 4
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48; // 48 * 4

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_safe_min, VF_QBSA_SAFE_PAD_VALUE);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_empty_lse, VF_QBSA_EMPTY_LSE_VALUE);
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_135);

    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * mConst, vreg_safe_min,
                                                          preg_135);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_135);
    for (uint64_t iter_m = 0; iter_m < uint64_t(ubN >> 2); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m4Const);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        if constexpr (needAtten) {
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * mConst);
            Select(vreg_data_tmp0, vreg_data_tmp0, vreg_min, preg_compare0);
            Select(vreg_data_tmp1, vreg_data_tmp1, vreg_min, preg_compare1);
            Select(vreg_data_tmp2, vreg_data_tmp2, vreg_min, preg_compare2);
            Select(vreg_data_tmp3, vreg_data_tmp3, vreg_min, preg_compare3);
        }

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m4Const, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m4Const, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m4Const, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m4Const, vreg_data_tmp3, preg_135);
        Max(max0, max0, vreg_data_tmp0, preg_135);
        Max(max1, max1, vreg_data_tmp1, preg_135);
        Max(max2, max2, vreg_data_tmp2, preg_135);
        Max(max3, max3, vreg_data_tmp3, preg_135);
    }

    // 第二个 sparse block 不存在时，尾部列可能对应零 scale。
    // q/k scale 处理后重新写入 minValue，避免无效列进入 softmax。
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * mConst, vreg_min,
                                                          preg_135);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::NE>(preg_invalid_cur, max0, minValue, preg_108);

    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Select(max0, max0, vreg_empty_lse, preg_invalid_cur);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m2Const);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const);

        LoadAlign(vreg_x_f32_4,
                  input_x_local_UB + i0 * m2Const + mConst); // 64 表示元素个数，取到了vreg_x_f32_0的下一行
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const + mConst);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_invalid_cur);

        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
        if constexpr (!IsSameType<T2, hifloat8_t>::value) {
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        }

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint32_t ubN = 128>
__aicore__ inline void ProcessVec1DnNoUpdatePerTokenHead(
    const LocalTensor<T2> &dstTensor, const LocalTensor<T> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t> &maskTensor, const LocalTensor<T> &qScaleTensor, const LocalTensor<T> &kScaleTensor,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, const T minValue, bool needAtten,
    const float pScale = 1.0f)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ float *qScaleUb = (__ubuf__ T *)qScaleTensor.GetPhyAddr();
    __ubuf__ float *kScaleUb = (__ubuf__ T *)kScaleTensor.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    dScale = scale;
    blockStride = ubN >> 2 | 0x1; // ubN / 4 + 1，表示搬运到UB上的时候，每块会间隔32bit，避免bank冲突
    repeatStride = 2;
    indexesUb = (__ubuf__ uint8_t *)vselrIndexesBuf.GetPhyAddr();

    if (needAtten && originN < ubN) {
        ProcessVec1DnNoUpdatePerTokenHeadAttenTailVF<T, T2, hasAtten, true, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale);
    } else if (originN < ubN) {
        ProcessVec1DnNoUpdatePerTokenHeadAttenTailVF<T, T2, hasAtten, false, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale);
    } else if (needAtten) {
        ProcessVec1DnNoUpdatePerTokenHeadVF<T, T2, hasAtten, true, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale);
    } else {
        ProcessVec1DnNoUpdatePerTokenHeadVF<T, T2, hasAtten, false, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale);
    }
}

template <typename T, typename T2, bool hasAtten = false, bool needAtten = false, uint32_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdatePerTokenHeadVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
    __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb, __ubuf__ float *qScaleUb, __ubuf__ float *kScaleUb,
    __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
    const T minValue, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f)
{
    constexpr uint32_t mConst = 64;                                                        // m = 64
    constexpr uint32_t m2Const = 128;                                                      // m * 2 = 128
    constexpr uint32_t m3Const = 192;                                                      // m * 3 = 192
    constexpr uint32_t m4Const = 256;                                                      // m * 4 = 256
    constexpr uint32_t ubN_m_m_div_4 = (ubN * mConst) >> 2;                                // ubN * m / 4
    constexpr uint32_t ubN_m_m_div_2 = (ubN * mConst) >> 1;                                // ubN * m / 2
    constexpr uint32_t ubN_m_m_mul3_div_4 = ((ubN * mConst) >> 1) + ((ubN * mConst) >> 2); // ubN * m * 3 / 4
    constexpr uint16_t loopNum = ubN >> 3;                                                 // ubN / 8

    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_empty_lse;
    MaskReg preg_invalid_cur;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    x_exp_1 = x_exp + 32; // Cast 之后第二个寄存器搬运到UB上的目标地址需偏移32字节

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + mConst;
    __ubuf__ float *src_ub2 = src_ub0 + m2Const;
    __ubuf__ float *src_ub3 = src_ub0 + m3Const;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16; // 16 * 4 个元素个数(attenMaskUb 的数据类型为uint8，这里是uint32)
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32; // 32 * 4
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48; // 48 * 4

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_empty_lse, VF_QBSA_EMPTY_LSE_VALUE);
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_135);

    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_135);
    for (uint64_t iter_m = 0; iter_m < uint64_t(ubN >> 2); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m4Const);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        if constexpr (needAtten) {
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * mConst);
            Select(vreg_data_tmp0, vreg_data_tmp0, vreg_min, preg_compare0);
            Select(vreg_data_tmp1, vreg_data_tmp1, vreg_min, preg_compare1);
            Select(vreg_data_tmp2, vreg_data_tmp2, vreg_min, preg_compare2);
            Select(vreg_data_tmp3, vreg_data_tmp3, vreg_min, preg_compare3);
        }

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m4Const, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m4Const, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m4Const, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m4Const, vreg_data_tmp3, preg_135);
        Max(max0, max0, vreg_data_tmp0, preg_135);
        Max(max1, max1, vreg_data_tmp1, preg_135);
        Max(max2, max2, vreg_data_tmp2, preg_135);
        Max(max3, max3, vreg_data_tmp3, preg_135);
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::NE>(preg_invalid_cur, max0, minValue, preg_108);

    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Select(max0, max0, vreg_empty_lse, preg_invalid_cur);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m2Const);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const);

        LoadAlign(vreg_x_f32_4,
                  input_x_local_UB + i0 * m2Const + mConst); // 64 表示元素个数，取到了vreg_x_f32_0的下一行
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const + mConst);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_invalid_cur);

        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
        if constexpr (!IsSameType<T2, hifloat8_t>::value) {
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        }

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    RegTensor<float> vreg_l0;
    MaskReg preg_previous_sum_zero;
    LoadAlign(vreg_l0, new_global_sum);
    Compare<float, CMPMODE::EQ>(preg_previous_sum_zero, vreg_l0, vreg_zero, preg_134);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Select(vreg_l0, vreg_zero, vreg_l0, preg_previous_sum_zero);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, bool needAtten = false, uint32_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdatePerTokenHeadAttenTailVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
    __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb, __ubuf__ float *qScaleUb, __ubuf__ float *kScaleUb,
    __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
    const T minValue, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f)
{
    constexpr uint32_t mConst = 64;                                                        // m = 64
    constexpr uint32_t m2Const = 128;                                                      // m * 2 = 128
    constexpr uint32_t m3Const = 192;                                                      // m * 3 = 192
    constexpr uint32_t m4Const = 256;                                                      // m * 4 = 256
    constexpr uint32_t ubN_m_m_div_4 = (ubN * mConst) >> 2;                                // ubN * m / 4
    constexpr uint32_t ubN_m_m_div_2 = (ubN * mConst) >> 1;                                // ubN * m / 2
    constexpr uint32_t ubN_m_m_mul3_div_4 = ((ubN * mConst) >> 1) + ((ubN * mConst) >> 2); // ubN * m * 3 / 4
    constexpr uint16_t loopNum = ubN >> 3;                                                 // ubN / 8

    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_safe_min;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_empty_lse;
    MaskReg preg_invalid_cur;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    x_exp_1 = x_exp + 32; // Cast 之后第二个寄存器搬运到UB上的目标地址需偏移32字节

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + mConst;
    __ubuf__ float *src_ub2 = src_ub0 + m2Const;
    __ubuf__ float *src_ub3 = src_ub0 + m3Const;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16; // 16 * 4 个元素个数(attenMaskUb 的数据类型为uint8，这里是uint32)
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32; // 32 * 4
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48; // 48 * 4

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_safe_min, VF_QBSA_SAFE_PAD_VALUE);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_empty_lse, VF_QBSA_EMPTY_LSE_VALUE);
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_135);

    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * mConst, vreg_safe_min,
                                                          preg_135);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_135);
    for (uint64_t iter_m = 0; iter_m < uint64_t(ubN >> 2); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m4Const);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m4Const);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        if constexpr (needAtten) {
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * mConst);
            Select(vreg_data_tmp0, vreg_data_tmp0, vreg_min, preg_compare0);
            Select(vreg_data_tmp1, vreg_data_tmp1, vreg_min, preg_compare1);
            Select(vreg_data_tmp2, vreg_data_tmp2, vreg_min, preg_compare2);
            Select(vreg_data_tmp3, vreg_data_tmp3, vreg_min, preg_compare3);
        }

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m4Const, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m4Const, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m4Const, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m4Const, vreg_data_tmp3, preg_135);
        Max(max0, max0, vreg_data_tmp0, preg_135);
        Max(max1, max1, vreg_data_tmp1, preg_135);
        Max(max2, max2, vreg_data_tmp2, preg_135);
        Max(max3, max3, vreg_data_tmp3, preg_135);
    }

    // Tail columns may have zero scale when the second sparse block is absent.
    // Re-apply minValue after q/k scale so invalid columns cannot enter softmax.
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * mConst, vreg_min,
                                                          preg_135);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::NE>(preg_invalid_cur, max0, minValue, preg_108);

    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Select(max0, max0, vreg_empty_lse, preg_invalid_cur);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m2Const);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const);

        LoadAlign(vreg_x_f32_4,
                  input_x_local_UB + i0 * m2Const + mConst); // 64 表示元素个数，取到了vreg_x_f32_0的下一行
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const + mConst);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_invalid_cur);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_invalid_cur);

        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
        if constexpr (!IsSameType<T2, hifloat8_t>::value) {
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
            Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        }

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    RegTensor<float> vreg_l0;
    MaskReg preg_previous_sum_zero;
    LoadAlign(vreg_l0, new_global_sum);
    Compare<float, CMPMODE::EQ>(preg_previous_sum_zero, vreg_l0, vreg_zero, preg_134);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Select(vreg_l0, vreg_zero, vreg_l0, preg_previous_sum_zero);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint32_t ubN = 128>
__aicore__ inline void ProcessVec1DnUpdatePerTokenHead(
    const LocalTensor<T2> &dstTensor, const LocalTensor<T> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t> &maskTensor, const LocalTensor<T> &qScaleTensor, const LocalTensor<T> &kScaleTensor,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, const T minValue, bool needAtten,
    const float pScale = 1.0f)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ float *qScaleUb = (__ubuf__ T *)qScaleTensor.GetPhyAddr();
    __ubuf__ float *kScaleUb = (__ubuf__ T *)kScaleTensor.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    dScale = scale;
    blockStride = ubN >> 2 | 0x1; // ubN / 4 + 1，表示搬运到UB上的时候，每块会间隔32bit，避免bank冲突
    repeatStride = 2;
    indexesUb = (__ubuf__ uint8_t *)vselrIndexesBuf.GetPhyAddr();

    if (needAtten && originN < ubN) {
        ProcessVec1DnUpdatePerTokenHeadAttenTailVF<T, T2, hasAtten, true, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale);
    } else if (originN < ubN) {
        ProcessVec1DnUpdatePerTokenHeadAttenTailVF<T, T2, hasAtten, false, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale);
    } else if (needAtten) {
        ProcessVec1DnUpdatePerTokenHeadVF<T, T2, hasAtten, true, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale);
    } else {
        ProcessVec1DnUpdatePerTokenHeadVF<T, T2, hasAtten, false, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, qScaleUb, kScaleUb,
            indexesUb, m, n, originN, scale, minValue, dScale, blockStride, repeatStride, pScale);
    }
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint32_t ubN = 256>
__aicore__ inline void ProcessVec1VfDnPerTokenHead(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &expSumTensor,
                                                   const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor,
                                                   const LocalTensor<T> &expMaxTensor, TBuf<> *vselrIndexesBuf,
                                                   const LocalTensor<uint8_t> &maskTensor,
                                                   const LocalTensor<T> &qScaleTensor,
                                                   const LocalTensor<T> &kScaleTensor, const uint32_t m,
                                                   const uint32_t n, const uint32_t originN, const T scale,
                                                   const T minValue, bool needAtten, const float pScale = 1.0f)
{
    if constexpr (!isUpdate) {
        LocalTensor<uint8_t> indexesTensor;
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        ProcessVec1DnNoUpdatePerTokenHead<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, qScaleTensor,
            kScaleTensor, m, n, originN, scale, minValue, needAtten, pScale);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        ProcessVec1DnUpdatePerTokenHead<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, qScaleTensor,
            kScaleTensor, m, n, originN, scale, minValue, needAtten, pScale);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, uint32_t SUB_LOOP = 0U>
__simd_vf__ inline void ProcessVec1DnNoUpdateMxfp8VF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
                                                     __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
                                                     __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb,
                                                     __ubuf__ uint8_t *indexesUb, __ubuf__ fp8_e8m0_t *pScaleSubLoop0,
                                                     const uint32_t originN, const float dScale, float pScale,
                                                     const T minValue)
{
    static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
    constexpr uint32_t mConst = 64;                                                        // m = 64
    constexpr uint32_t m2Const = 128;                                                      // m * 2 = 128
    constexpr uint32_t m3Const = 192;                                                      // m * 3 = 192
    constexpr uint32_t m4Const = 256;                                                      // m * 4 = 256
    constexpr uint32_t ubN_m_m_div_4 = (ubN * mConst) >> 2;                                // ubN * m / 4
    constexpr uint32_t ubN_m_m_div_2 = (ubN * mConst) >> 1;                                // ubN * m / 2
    constexpr uint32_t ubN_m_m_mul3_div_4 = ((ubN * mConst) >> 1) + ((ubN * mConst) >> 2); // ubN * m * 3 / 4
    constexpr uint16_t loopNum = ubN >> 3;                                                 // ubN / 8
    constexpr uint32_t blockStride = (ubN >> 2U) | 0x1U;
    constexpr uint32_t repeatStride = 2U;
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_zero;
    MaskReg preg_invalid_cur;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    __ubuf__ T2 *x_exp_1;

    x_exp_1 = x_exp + (ubN >> 3);

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + mConst;
    __ubuf__ float *src_ub2 = src_ub0 + m2Const;
    __ubuf__ float *src_ub3 = src_ub0 + m3Const;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * mConst, vreg_min,
                                                          preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN >> 2); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m4Const);
            LoadAlign(src1, src_ub1 + iter_m * m4Const);
            LoadAlign(src2, src_ub2 + iter_m * m4Const);
            LoadAlign(src3, src_ub3 + iter_m * m4Const);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * mConst);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m4Const, src0, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m4Const, src1, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m4Const, src2, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m4Const, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN >> 2); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m4Const);
            LoadAlign(src1, src_ub1 + iter_m * m4Const);
            LoadAlign(src2, src_ub2 + iter_m * m4Const);
            LoadAlign(src3, src_ub3 + iter_m * m4Const);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::EQ>(preg_invalid_cur, max0, minValue, preg_108);
    Muls(max0, max0, dScale, preg_108);
    Muls(max0, max0, INV_LN2, preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, LN2, preg_108);
    Select(max0, vreg_min, max0, preg_invalid_cur);

    if constexpr (SUB_LOOP == 1U) {
        LoadAlign(vreg_x_max_f32_b, new_global_max);
        Max(max0, max0, vreg_x_max_f32_b, preg_108);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    }

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);

    if constexpr (hasAtten) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m2Const);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const);

        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const + mConst);

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);
        Select(vreg_x_exp_0, vreg_zero, vreg_x_exp_0, preg_invalid_cur);
        Select(vreg_x_exp_1, vreg_zero, vreg_x_exp_1, preg_invalid_cur);
        Select(vreg_x_exp_2, vreg_zero, vreg_x_exp_2, preg_invalid_cur);
        Select(vreg_x_exp_3, vreg_zero, vreg_x_exp_3, preg_invalid_cur);

        Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
        Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
        Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
        Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134); // vreg_x_exp_i softmax分子
        Select(vreg_x_exp_4, vreg_zero, vreg_x_exp_4, preg_invalid_cur);
        Select(vreg_x_exp_5, vreg_zero, vreg_x_exp_5, preg_invalid_cur);
        Select(vreg_x_exp_6, vreg_zero, vreg_x_exp_6, preg_invalid_cur);
        Select(vreg_x_exp_7, vreg_zero, vreg_x_exp_7, preg_invalid_cur);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        // Interleave two independent FP8 pack chains to cover Cast/Or/Gather RAW latency.
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);

    if constexpr (SUB_LOOP == 0U) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
        // pscale update
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
        Duplicate(vreg_p_scale_f8e8m0, 0x7f, preg_134);
        StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                                  vreg_p_scale_f8e8m0, preg_134);
    } else {
        RegTensor<float> first_loop_sum;
        LoadAlign(first_loop_sum, new_global_sum);
        Mul(first_loop_sum, vreg_x_max_f32_b, first_loop_sum, preg_134);
        Add(vreg_x_sum0, first_loop_sum, vreg_x_sum0, preg_134);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
        // pscale update
        RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
        RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
        Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_x_max_f32_b, preg_135);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_108);
        Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_x_max_f32_b, preg_135);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_108);
        Or((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0,
           (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, preg_134);
        DeInterleave(vreg_p_scale_f8e8m0_dst0, vreg_p_scale_f8e8m0_dst1, vreg_p_scale_f8e8m0_0, vreg_p_scale_f8e8m0_0);
        StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                                  vreg_p_scale_f8e8m0_dst0, preg_134);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, uint32_t SUB_LOOP = 0U>
__aicore__ inline void ProcessVec1DnNoUpdateMxfp8(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &expSumTensor,
                                                  const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor,
                                                  const LocalTensor<T> &expMaxTensor,
                                                  const LocalTensor<uint8_t> &vselrIndexesBuf,
                                                  const LocalTensor<uint8_t> &maskTensor, const uint32_t originN,
                                                  const float dScale, float pScale, const T minValue,
                                                  const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0Tensor.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesBuf.GetPhyAddr();

    ProcessVec1DnNoUpdateMxfp8VF<T, T2, hasAtten, ubN, SUB_LOOP>(x_exp, input_x_local_UB, exp_max_fp32, new_global_sum,
                                                                 new_global_max, maskUb, indexesUb, pScaleSubLoop0Ub,
                                                                 originN, dScale, pScale, minValue);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, uint32_t SUB_LOOP = 0U>
__simd_vf__ inline void ProcessVec1DnUpdateMxfp8VF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
                                                   __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum,
                                                   __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb,
                                                   __ubuf__ uint8_t *indexesUb, __ubuf__ fp8_e8m0_t *pScaleSubLoop0,
                                                   const uint32_t originN, const float dScale, float pScale,
                                                   const T minValue, __ubuf__ float *pre_loop_max,
                                                   __ubuf__ float *pre_loop_sum, __ubuf__ float *first_loop_sum)
{
    static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
    constexpr uint32_t mConst = 64;                                                        // m = 64
    constexpr uint32_t m2Const = 128;                                                      // m * 2 = 128
    constexpr uint32_t m3Const = 192;                                                      // m * 3 = 192
    constexpr uint32_t m4Const = 256;                                                      // m * 4 = 256
    constexpr uint32_t ubN_m_m_div_4 = (ubN * mConst) >> 2;                                // ubN * m / 4
    constexpr uint32_t ubN_m_m_div_2 = (ubN * mConst) >> 1;                                // ubN * m / 2
    constexpr uint32_t ubN_m_m_mul3_div_4 = ((ubN * mConst) >> 1) + ((ubN * mConst) >> 2); // ubN * m * 3 / 4
    constexpr uint16_t loopNum = ubN >> 3;
    constexpr uint32_t blockStride = (ubN >> 2U) | 0x1U;
    constexpr uint32_t repeatStride = 2U;
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<float> vreg_subloop_update;
    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_zero;
    MaskReg preg_invalid_cur;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    x_exp_1 = x_exp + (ubN >> 3);
    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + mConst;
    __ubuf__ float *src_ub2 = src_ub0 + m2Const;
    __ubuf__ float *src_ub3 = src_ub0 + m3Const;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * mConst, vreg_min,
                                                          preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN >> 2); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m4Const);
            LoadAlign(src1, src_ub1 + iter_m * m4Const);
            LoadAlign(src2, src_ub2 + iter_m * m4Const);
            LoadAlign(src3, src_ub3 + iter_m * m4Const);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * mConst);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * mConst);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m4Const, src0, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m4Const, src1, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m4Const, src2, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m4Const, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN >> 2); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m4Const);
            LoadAlign(src1, src_ub1 + iter_m * m4Const);
            LoadAlign(src2, src_ub2 + iter_m * m4Const);
            LoadAlign(src3, src_ub3 + iter_m * m4Const);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Compares<T, CMPMODE::EQ>(preg_invalid_cur, max0, minValue, preg_108);
    Muls(max0, max0, dScale, preg_108);
    Muls(max0, max0, INV_LN2, preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, LN2, preg_108);
    Select(max0, vreg_min, max0, preg_invalid_cur);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    if constexpr (SUB_LOOP == 0U) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)pre_loop_max, vreg_x_max_f32_b, preg_108);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    } else {
        FusedExpSub(vreg_subloop_update, vreg_x_max_f32_b, max0, preg_134);
        LoadAlign(vreg_x_max_f32_b, pre_loop_max);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    }

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);

    if constexpr (hasAtten) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m2Const);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const);

        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN_m_m_div_4 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN_m_m_div_2 + i0 * m2Const + mConst);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN_m_m_mul3_div_4 + i0 * m2Const + mConst);

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);
        Select(vreg_x_exp_0, vreg_zero, vreg_x_exp_0, preg_invalid_cur);
        Select(vreg_x_exp_1, vreg_zero, vreg_x_exp_1, preg_invalid_cur);
        Select(vreg_x_exp_2, vreg_zero, vreg_x_exp_2, preg_invalid_cur);
        Select(vreg_x_exp_3, vreg_zero, vreg_x_exp_3, preg_invalid_cur);

        Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
        Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
        Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
        Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        Select(vreg_x_exp_4, vreg_zero, vreg_x_exp_4, preg_invalid_cur);
        Select(vreg_x_exp_5, vreg_zero, vreg_x_exp_5, preg_invalid_cur);
        Select(vreg_x_exp_6, vreg_zero, vreg_x_exp_6, preg_invalid_cur);
        Select(vreg_x_exp_7, vreg_zero, vreg_x_exp_7, preg_invalid_cur);

        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        // Interleave two independent FP8 pack chains to cover Cast/Or/Gather RAW latency.
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Cast<T2, T, castTraitRintTwo>((RegTensor<T2> &)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Cast<T2, T, castTraitRintThree>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);

        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);

    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);

    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);

    RegTensor<float> vreg_l0;
    if constexpr (SUB_LOOP == 0U) {
        LoadAlign(vreg_l0, new_global_sum);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)pre_loop_sum, vreg_l0, preg_134);
        Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
        Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)first_loop_sum, vreg_x_sum0, preg_134);
        // pscale update
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
        Duplicate(vreg_p_scale_f8e8m0, 0x7f, preg_134);
        StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                                  vreg_p_scale_f8e8m0, preg_134);
    } else {
        RegTensor<float> vreg_l1;
        LoadAlign(vreg_l0, first_loop_sum);
        LoadAlign(vreg_l1, pre_loop_sum);
        Mul(vreg_l0, vreg_subloop_update, vreg_l0, preg_134);
        Add(vreg_x_sum0, vreg_l0, vreg_x_sum0, preg_134);
        Mul(vreg_l1, vreg_x_max_f32_b, vreg_l1, preg_134);
        Add(vreg_l0, vreg_x_sum0, vreg_l1, preg_134);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
        // pscale update
        RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
        RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
        Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_subloop_update, preg_135);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_108);
        Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_subloop_update, preg_135);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_108);
        Or((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0,
           (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, preg_134);
        DeInterleave(vreg_p_scale_f8e8m0_dst0, vreg_p_scale_f8e8m0_dst1, vreg_p_scale_f8e8m0_0, vreg_p_scale_f8e8m0_0);
        StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                                  vreg_p_scale_f8e8m0_dst0, preg_134);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, uint32_t SUB_LOOP = 0U>
__aicore__ inline void ProcessVec1DnUpdateMxfp8(
    const LocalTensor<T2> &dstTensor, const LocalTensor<T> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t> &maskTensor, const uint32_t originN, const float dScale, float pScale, const T minValue,
    const LocalTensor<float> &preLoopMaxTensor, const LocalTensor<float> &preLoopSumTensor,
    const LocalTensor<float> &firstLoopSumTensor, const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ float *pre_loop_max = (__ubuf__ T *)preLoopMaxTensor.GetPhyAddr();
    __ubuf__ float *pre_loop_sum = (__ubuf__ T *)preLoopSumTensor.GetPhyAddr();
    __ubuf__ float *first_loop_sum = (__ubuf__ T *)firstLoopSumTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0Tensor.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesBuf.GetPhyAddr();

    ProcessVec1DnUpdateMxfp8VF<T, T2, hasAtten, ubN, SUB_LOOP>(
        x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, indexesUb, pScaleSubLoop0Ub,
        originN, dScale, pScale, minValue, pre_loop_max, pre_loop_sum, first_loop_sum);
}

/*
 * @ingroup ProcessVec1Vf
 * @brief 计算 max = reducemax 以及 exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor 输出 LocalTensor
 * @param [out] expSumTensor 最后一维 sum(exp(x-max)) 输出
 * @param [out] maxTensor 最后一维 max 输出
 * @param [in] srcTensor 输入 LocalTensor
 * @param [out] expMaxTensor expmax 输出 LocalTensor
 * @param [in] sharedTmpBuffer 本地临时 Tensor
 * @param [in] originN 原始输入列数，支持范围为 0 < originN <= 128
 * @param [in] dScale QK 反量化与 softmax scale 的乘积
 * @param [in] minValue 最小值
 * @param [in] isUpdate 是否启用 flash update 模式
 * @param [in] oriNRange originN 范围
 */

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256,
          uint32_t SUB_LOOP = 0U>
__aicore__ inline void ProcessVec1VfDnMxfp8(
    const LocalTensor<T2> &dstTensor, const LocalTensor<T> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor, TBuf<> *vselrIndexesBuf,
    const LocalTensor<uint8_t> &maskTensor, const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor, const uint32_t originN,
    const float dScale, float pScale, const T minValue, const LocalTensor<T> &preLoopMaxTensor,
    const LocalTensor<T> &preLoopSumTensor, const LocalTensor<T> &firstLoopSumTensor)
{
    // 将 256-column C1 转为 P(e4m3)/PScale(e8m0)；isUpdate 控制 online-softmax 更新。
    LocalTensor<uint8_t> indexesTensor;
    indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdateMxfp8<T, T2, hasAtten, ubN, SUB_LOOP>(dstTensor, expSumTensor, maxTensor, srcTensor,
                                                                   expMaxTensor, indexesTensor, maskTensor, originN,
                                                                   dScale, pScale, minValue, pScaleSubLoop0Tensor);
    } else {
        ProcessVec1DnUpdateMxfp8<T, T2, hasAtten, ubN, SUB_LOOP>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, originN, dScale,
            pScale, minValue, preLoopMaxTensor, preLoopSumTensor, firstLoopSumTensor, pScaleSubLoop0Tensor);
    }
}

template <typename T>
__simd_vf__ inline void BroadCastMaxSumVF(__ubuf__ float *out_ub, __ubuf__ float *ori_ub, const uint16_t loopM)
{
    RegTensor<float> broadcast_reg;
    MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
    for (uint16_t i = 0; i < loopM; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_E2B_B32>(broadcast_reg, ori_ub + i * 8);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)out_ub + i * 64, broadcast_reg, preg_all);
    }
}

template <typename T>
__aicore__ inline void BroadcastMaxSum(const LocalTensor<T> &outTensor, const LocalTensor<T> &oriTensor,
                                       uint32_t vecS1RealSize)
{
    __ubuf__ float *out_ub = (__ubuf__ T *)outTensor.GetPhyAddr();
    __ubuf__ float *ori_ub = (__ubuf__ T *)oriTensor.GetPhyAddr();

    // Align8, broadcast one element to 8 elements, one register can store 64 elements,
    // so we can handle 64 / 8 = 8 elements per loop.
    uint16_t loopM = (vecS1RealSize + 7) >> 3;
    BroadCastMaxSumVF<T>(out_ub, ori_ub, loopM);
}
} // namespace FaVectorApi
#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
