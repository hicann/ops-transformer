/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root.
 */
#ifndef QBSA_MX_M128_V1_FULL_N_H_
#define QBSA_MX_M128_V1_FULL_N_H_
#include "../common/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"

// [M128_V1_FULL_N] Isolated f5067f80a VF copies. Only omit the empty padding loop.
// Keep every mask/math/packing operation and both memory-ordering barriers.
#ifndef QBSA_9102_MX_M128_V1_FULL_N
#define QBSA_9102_MX_M128_V1_FULL_N 1
#endif
static_assert(QBSA_9102_MX_M128_V1_FULL_N == 0 || QBSA_9102_MX_M128_V1_FULL_N == 1,
              "QBSA_9102_MX_M128_V1_FULL_N must be 0 or 1");
namespace FaVectorApi {
#if QBSA_9102_MX_M128_V1_FULL_N
template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, uint32_t SUB_LOOP = 0U>
__simd_vf__ inline void ProcessVec1DnNoUpdateMxfp8FullNVF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                          __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                          __ubuf__ float* new_global_max, __ubuf__ uint32_t* maskUb,
                                                          __ubuf__ uint8_t* indexesUb,
                                                          __ubuf__ fp8_e8m0_t* pScaleSubLoop0, const uint32_t originN,
                                                          const float dScale, float pScale, const T minValue)
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
    MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_zero;
    MaskReg preg_invalid_cur;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ uint32_t* mask_ub0 = maskUb;
    __ubuf__ uint32_t* mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t* mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t* mask_ub3 = maskUb + 48;

    __ubuf__ T2* x_exp_1;

    x_exp_1 = x_exp + (ubN >> 3);

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + mConst;
    __ubuf__ float* src_ub2 = src_ub0 + m2Const;
    __ubuf__ float* src_ub3 = src_ub0 + m3Const;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    // [M128_V1_FULL_N] Dispatched only for originN == ubN == 256.
    // No padding store or runtime tail loop. Keep the cross-VF VST/VLD ordering.
    static_assert(ubN == 256U, "Full-N VF is specialized for 256 columns");
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN >> 2); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m4Const);
            LoadAlign(src1, src_ub1 + iter_m * m4Const);
            LoadAlign(src2, src_ub2 + iter_m * m4Const);
            LoadAlign(src3, src_ub3 + iter_m * m4Const);
            LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * mConst);
            LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * mConst);
            LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * mConst);
            LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * mConst);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m4Const, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m4Const, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m4Const, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m4Const, src3, preg_108);
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

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
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
        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);

        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
    Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);

    if constexpr (SUB_LOOP == 0U) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum0, preg_134);
        // pscale update
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
        Duplicate(vreg_p_scale_f8e8m0, 0x7f, preg_134);
        StoreAlign<fp8_e8m0_t, Reg::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t*&)pScaleSubLoop0),
                                                             vreg_p_scale_f8e8m0, preg_134);
    } else {
        RegTensor<float> first_loop_sum;
        LoadAlign(first_loop_sum, new_global_sum);
        Mul(first_loop_sum, vreg_x_max_f32_b, first_loop_sum, preg_134);
        Add(vreg_x_sum0, first_loop_sum, vreg_x_sum0, preg_134);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum0, preg_134);
        // pscale update
        RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
        RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
        Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_x_max_f32_b, preg_135);
#if QBSA_ARCH_9102R1
        EncodeMxPScaleE8M0Zero(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_108);
#else
        Cast<fp8_e8m0_t, bfloat16_t, castTraitE8M0Zero>(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_108);
#endif
        Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_x_max_f32_b, preg_135);
#if QBSA_ARCH_9102R1
        EncodeMxPScaleE8M0Zero(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_108);
#else
        Cast<fp8_e8m0_t, bfloat16_t, castTraitE8M0Zero>(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_108);
#endif
        Or((RegTensor<uint8_t>&)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t>&)vreg_p_scale_f8e8m0_0,
           (RegTensor<uint8_t>&)vreg_p_scale_f8e8m0_1, preg_134);
        DeInterleave(vreg_p_scale_f8e8m0_dst0, vreg_p_scale_f8e8m0_dst1, vreg_p_scale_f8e8m0_0, vreg_p_scale_f8e8m0_0);
        StoreAlign<fp8_e8m0_t, Reg::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t*&)pScaleSubLoop0),
                                                             vreg_p_scale_f8e8m0_dst0, preg_134);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, uint32_t SUB_LOOP = 0U,
          bool USE_RECURSIVE_SUM = false>
__simd_vf__ inline void ProcessVec1DnUpdateMxfp8FullNVF(
    __ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB, __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
    __ubuf__ float* new_global_max, __ubuf__ uint32_t* maskUb, __ubuf__ uint8_t* indexesUb,
    __ubuf__ fp8_e8m0_t* pScaleSubLoop0, const uint32_t originN, const float dScale, float pScale, const T minValue,
    __ubuf__ float* pre_loop_max, __ubuf__ float* pre_loop_sum, __ubuf__ float* first_loop_sum)
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
    MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_zero;
    MaskReg preg_invalid_cur;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2* x_exp_1;
    x_exp_1 = x_exp + (ubN >> 3);
    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + mConst;
    __ubuf__ float* src_ub2 = src_ub0 + m2Const;
    __ubuf__ float* src_ub3 = src_ub0 + m3Const;
    __ubuf__ uint32_t* mask_ub0 = maskUb;
    __ubuf__ uint32_t* mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t* mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t* mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_zero, 0.0f);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    // [M128_V1_FULL_N] Dispatched only for originN == ubN == 256.
    // No padding store or runtime tail loop. Keep the cross-VF VST/VLD ordering.
    static_assert(ubN == 256U, "Full-N VF is specialized for 256 columns");
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN >> 2); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m4Const);
            LoadAlign(src1, src_ub1 + iter_m * m4Const);
            LoadAlign(src2, src_ub2 + iter_m * m4Const);
            LoadAlign(src3, src_ub3 + iter_m * m4Const);
            LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * mConst);
            LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * mConst);
            LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * mConst);
            LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * mConst);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m4Const, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m4Const, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m4Const, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m4Const, src3, preg_108);
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
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)pre_loop_max, vreg_x_max_f32_b, preg_108);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    } else {
        FusedExpSub(vreg_subloop_update, vreg_x_max_f32_b, max0, preg_134);
        LoadAlign(vreg_x_max_f32_b, pre_loop_max);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    }

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
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
        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);

        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
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
        if constexpr (!USE_RECURSIVE_SUM) {
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)pre_loop_sum, vreg_l0, preg_134);
        }
        Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
        Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
        if constexpr (!USE_RECURSIVE_SUM) {
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)first_loop_sum, vreg_x_sum0, preg_134);
        }
        // pscale update
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
        Duplicate(vreg_p_scale_f8e8m0, 0x7f, preg_134);
        StoreAlign<fp8_e8m0_t, Reg::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t*&)pScaleSubLoop0),
                                                             vreg_p_scale_f8e8m0, preg_134);
    } else {
        if constexpr (USE_RECURSIVE_SUM) {
            // subLoop0 已将 Sprev * exp(Mprev - Mfirst) + S0 写回；这里只需随新 max 再缩放并累加 S1。
            LoadAlign(vreg_l0, new_global_sum);
            Mul(vreg_l0, vreg_subloop_update, vreg_l0, preg_134);
            Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
        } else {
            RegTensor<float> vreg_l1;
            LoadAlign(vreg_l0, first_loop_sum);
            LoadAlign(vreg_l1, pre_loop_sum);
            Mul(vreg_l0, vreg_subloop_update, vreg_l0, preg_134);
            Add(vreg_x_sum0, vreg_l0, vreg_x_sum0, preg_134);
            Mul(vreg_l1, vreg_x_max_f32_b, vreg_l1, preg_134);
            Add(vreg_l0, vreg_x_sum0, vreg_l1, preg_134);
        }
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
        // pscale update
        RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
        RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
        Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_subloop_update, preg_135);
#if QBSA_ARCH_9102R1
        EncodeMxPScaleE8M0Zero(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_108);
#else
        Cast<fp8_e8m0_t, bfloat16_t, castTraitE8M0Zero>(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_108);
#endif
        Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_subloop_update, preg_135);
#if QBSA_ARCH_9102R1
        EncodeMxPScaleE8M0Zero(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_108);
#else
        Cast<fp8_e8m0_t, bfloat16_t, castTraitE8M0Zero>(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_108);
#endif
        Or((RegTensor<uint8_t>&)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t>&)vreg_p_scale_f8e8m0_0,
           (RegTensor<uint8_t>&)vreg_p_scale_f8e8m0_1, preg_134);
        DeInterleave(vreg_p_scale_f8e8m0_dst0, vreg_p_scale_f8e8m0_dst1, vreg_p_scale_f8e8m0_0, vreg_p_scale_f8e8m0_0);
        StoreAlign<fp8_e8m0_t, Reg::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t*&)pScaleSubLoop0),
                                                             vreg_p_scale_f8e8m0_dst0, preg_134);
    }
}
#endif

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256,
          uint32_t SUB_LOOP = 0U>
__aicore__ inline void ProcessVec1VfDnMxfp8RecursiveSumFullN(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, TBuf<>* vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<fp8_e8m0_t>& pScaleSubLoop0Tensor, const uint32_t originN,
    const float dScale, float pScale, const T minValue, const LocalTensor<T>& preLoopMaxTensor)
{
#if QBSA_9102_MX_M128_V1_FULL_N
    if constexpr (ubN == 256U) {
        if (originN == ubN) {
            LocalTensor<uint8_t> indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
            __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
            __ubuf__ float* input_x = (__ubuf__ T*)srcTensor.GetPhyAddr();
            __ubuf__ float* exp_max = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
            __ubuf__ float* sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
            __ubuf__ float* max = (__ubuf__ T*)maxTensor.GetPhyAddr();
            __ubuf__ uint32_t* mask = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
            __ubuf__ uint8_t* indexes = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
            __ubuf__ fp8_e8m0_t* scale = (__ubuf__ fp8_e8m0_t*)pScaleSubLoop0Tensor.GetPhyAddr();
            if constexpr (!isUpdate) {
                ProcessVec1DnNoUpdateMxfp8FullNVF<T, T2, hasAtten, ubN, SUB_LOOP>(
                    x_exp, input_x, exp_max, sum, max, mask, indexes, scale, originN, dScale, pScale, minValue);
            } else {
                __ubuf__ float* pre_max = (__ubuf__ T*)preLoopMaxTensor.GetPhyAddr();
                ProcessVec1DnUpdateMxfp8FullNVF<T, T2, hasAtten, ubN, SUB_LOOP, true>(
                    x_exp, input_x, exp_max, sum, max, mask, indexes, scale, originN, dScale, pScale, minValue, pre_max,
                    pre_max, pre_max);
            }
            return;
        }
    }
#endif
    ProcessVec1VfDnMxfp8RecursiveSum<T, T2, isUpdate, hasAtten, ubN, SUB_LOOP>(
        dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, vselrIndexesBuf, maskTensor, pScaleSubLoop0Tensor,
        originN, dScale, pScale, minValue, preLoopMaxTensor);
}

} // namespace FaVectorApi
#endif
