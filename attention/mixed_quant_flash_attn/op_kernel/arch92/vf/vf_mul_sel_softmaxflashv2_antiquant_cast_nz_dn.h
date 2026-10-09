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
 * \file vf_mul_sel_softmaxflashv2_antiquant_cast_nz_dn.h
 * \brief
 */

#ifndef MUL_SEL_SOFTMAXFLASHV2_ANTIQUANT_CAST_NZ_DN_H_
#define MUL_SEL_SOFTMAXFLASHV2_ANTIQUANT_CAST_NZ_DN_H_
#include "kernel_tensor.h"
namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false

template <typename T, typename T2, bool hasAtten = false, uint16_t s1BaseSize = 128, uint16_t ubN = 128,
          bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnNoUpdate64VF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                  __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                  __ubuf__ float* new_global_max, __ubuf__ uint32_t* maskUb,
                                                  __ubuf__ uint8_t* indexesUb, const uint32_t m, const uint32_t n,
                                                  const uint32_t originN, const T scale, float deScaleQK,
                                                  const T minValue, float keepProb, bool needAtten, const float dScale,
                                                  const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    constexpr uint16_t repeatStride = 1;
    const uint16_t blockStride = (ubN >> 1) | 0x1;

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
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;
    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t* mask_ub0 = maskUb;
    __ubuf__ uint32_t* mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t* mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t* mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);

    if constexpr (hasSink) {
        Max(max0, max0, vreg_sink_input, preg_108);
    }

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
            Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
            Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
            Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }

        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1,
                         vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                             IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t s1BaseSize = 128, uint16_t ubN = 128,
          bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnNoUpdate32VF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                  __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                  __ubuf__ float* new_global_max, __ubuf__ uint32_t* maskUb,
                                                  __ubuf__ uint8_t* indexesUb, const uint32_t m, const uint32_t n,
                                                  const uint32_t originN, const T scale, float deScaleQK,
                                                  const T minValue, float keepProb, bool needAtten, const float dScale,
                                                  const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    constexpr uint16_t repeatStride = 1;
    constexpr uint16_t packCols = 4u;
    const uint16_t blockStride = ubN / packCols | 0x1;
    const uint32_t maskColStride = m >> 2;
    __ubuf__ T* src0_ub = (__ubuf__ T*)input_x_local_UB;
    __ubuf__ T* src0_ub_1 = (__ubuf__ T*)input_x_local_UB + m;
    __ubuf__ T* src0_ub1 = src0_ub + m * 2;
    __ubuf__ T* src0_ub1_1 = src0_ub + m * 3;
    __ubuf__ T* src0_ub2 = src0_ub + m * 4;
    __ubuf__ T* src0_ub2_1 = src0_ub + m * 5;
    __ubuf__ T* src0_ub3 = src0_ub + m * 6;
    __ubuf__ T* src0_ub3_1 = src0_ub + m * 7;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vreg_x_sum_even;
        MicroAPI::RegTensor<T> vreg_x_sum_odd;
        MicroAPI::RegTensor<T> vreg_x_sum_1_even;
        MicroAPI::RegTensor<T> vreg_x_sum_1_odd;
        MicroAPI::RegTensor<T> vreg_x_sum0;
        MicroAPI::RegTensor<float> vreg_x_sum0_even;
        MicroAPI::RegTensor<float> vreg_x_sum0_odd;
        MicroAPI::RegTensor<T> vreg_x_sum1;
        MicroAPI::RegTensor<T2> vreg_x_exp_even_f16;
        MicroAPI::RegTensor<T2> vreg_x_exp_odd_f16;
        MicroAPI::RegTensor<T2> vreg_x_exp_even_bf16;
        MicroAPI::RegTensor<T2> vreg_x_exp_odd_bf16;

        MicroAPI::RegTensor<T> vreg_x_exp_even;
        MicroAPI::RegTensor<T> vreg_x_exp_odd;
        MicroAPI::RegTensor<T> vreg_x_f32_a;
        MicroAPI::RegTensor<T> vreg_x_f32_b;
        MicroAPI::RegTensor<T> vreg_x_exp_even_1;
        MicroAPI::RegTensor<T> vreg_x_exp_odd_1;
        MicroAPI::RegTensor<T2> vreg_x_exp_even_f16_1;
        MicroAPI::RegTensor<T2> vreg_x_exp_odd_f16_1;
        MicroAPI::RegTensor<T2> vreg_x_exp_even_bf16_1;
        MicroAPI::RegTensor<T2> vreg_x_exp_odd_bf16_1;

        MicroAPI::RegTensor<T> vreg_x_f32_1_a;
        MicroAPI::RegTensor<T> vreg_x_f32_1_b;
        MicroAPI::RegTensor<T2> vreg_x_exp_f16_pack;
        MicroAPI::RegTensor<T2> vreg_x_exp_f16_1_pack;
        MicroAPI::RegTensor<T2> vreg_x_exp_f16_packa;
        MicroAPI::RegTensor<T2> vreg_x_exp_f16_1_packa;
        MicroAPI::RegTensor<T2> vreg_x_exp_bf16_pack;
        MicroAPI::RegTensor<T2> vreg_x_exp_bf16_1_pack;
        MicroAPI::RegTensor<T2> vreg_x_exp_bf16_packa;
        MicroAPI::RegTensor<T2> vreg_x_exp_bf16_1_packa;
        MicroAPI::MaskReg preg_100;
        MicroAPI::MaskReg preg_101;
        MicroAPI::MaskReg preg_134;
        MicroAPI::MaskReg preg_135;
        MicroAPI::MaskReg preg_136;
        MicroAPI::MaskReg preg_all_b16;
        MicroAPI::MaskReg preg_all_b32;
        MicroAPI::MaskReg preg_all_s32;
        preg_134 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        preg_135 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        uint32_t sreg_92 = (uint32_t)128ULL;
        preg_136 = MicroAPI::UpdateMask<uint16_t>(sreg_92);
        preg_all_b16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        preg_all_b32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        preg_all_s32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_108;
        MicroAPI::MaskReg preg_compare0;
        MicroAPI::MaskReg preg_compare1;
        MicroAPI::MaskReg preg_compare2;
        MicroAPI::MaskReg preg_compare3;
        MicroAPI::RegTensor<int32_t> idx_swap;
        MicroAPI::RegTensor<int32_t> idx_swap16;
        MicroAPI::RegTensor<int32_t> idx_mask63;
        MicroAPI::RegTensor<T> src_00a, src_01a, src_02a, src_03a;
        MicroAPI::RegTensor<T> src_00b, src_01b, src_02b, src_03b;
        MicroAPI::RegTensor<T> src_10a, src_11a, src_12a, src_13a;
        MicroAPI::RegTensor<T> src_10b, src_11b, src_12b, src_13b;
        MicroAPI::RegTensor<T> max_0a, max_1a, max_2a, max_3a;
        MicroAPI::RegTensor<T> max_0b, max_1b, max_2b, max_3b;
        MicroAPI::RegTensor<T> vreg_min;

        MicroAPI::Duplicate<T, T>(max_0a, minValue);
        MicroAPI::Duplicate<T, T>(max_0b, minValue);
        MicroAPI::Duplicate<T, T>(max_1a, minValue);
        MicroAPI::Duplicate<T, T>(max_1b, minValue);
        MicroAPI::Duplicate<T, T>(max_2a, minValue);
        MicroAPI::Duplicate<T, T>(max_2b, minValue);
        MicroAPI::Duplicate<T, T>(max_3a, minValue);
        MicroAPI::Duplicate<T, T>(max_3b, minValue);
        MicroAPI::Duplicate<T, T>(vreg_min, minValue);

        const uint16_t tailBegin = static_cast<uint16_t>(originN);
        constexpr uint16_t tailStep = 2;
        constexpr uint16_t tailUnroll = 4u;
        constexpr uint16_t tailSpan = tailStep * tailUnroll;
        const uint16_t tailStoreCount =
            tailBegin < ubN ? static_cast<uint16_t>((ubN - tailBegin + tailStep - 1u) / tailStep) : 0u;
        if (tailStoreCount != 0u) {
            const uint16_t tailMainCount = static_cast<uint16_t>(tailStoreCount / tailUnroll);
            for (uint16_t j = 0u; j < tailMainCount; ++j) {
                uint16_t i = static_cast<uint16_t>(tailBegin + j * tailSpan);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*)input_x_local_UB + i * m,
                                                                          vreg_min, preg_135);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*)input_x_local_UB + (i + 2u) * m,
                                                                          vreg_min, preg_135);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*)input_x_local_UB + (i + 4u) * m,
                                                                          vreg_min, preg_135);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*)input_x_local_UB + (i + 6u) * m,
                                                                          vreg_min, preg_135);
            }
            const uint16_t tailMainEnd = static_cast<uint16_t>(tailBegin + tailMainCount * tailSpan);
            const uint16_t tailRemainCount = static_cast<uint16_t>(tailStoreCount - tailMainCount * tailUnroll);
            for (uint16_t j = 0u; j < tailRemainCount; ++j) {
                uint16_t i = static_cast<uint16_t>(tailMainEnd + j * tailStep);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*)input_x_local_UB + i * m,
                                                                          vreg_min, preg_135);
            }
            mem_bar(VST_VLD);
        }

        preg_108 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange<int32_t>(idx_swap16, static_cast<int32_t>(16));
        MicroAPI::Arange<int32_t>(idx_swap, static_cast<int32_t>(32));
        MicroAPI::Duplicate<int32_t, int32_t>(idx_mask63, static_cast<int32_t>(63));
        MicroAPI::And<int32_t, MicroAPI::MaskMergeMode::ZEROING>(idx_swap16, idx_swap16, idx_mask63, preg_all_s32);
        MicroAPI::And<int32_t, MicroAPI::MaskMergeMode::ZEROING>(idx_swap, idx_swap, idx_mask63, preg_all_s32);
        uint16_t maxLoopCount = static_cast<uint16_t>(ubN / 8u);

        if constexpr (hasAtten) {
            if (needAtten) {
                for (uint16_t iter_m = 0; iter_m < maxLoopCount; ++iter_m) {
                    const uint16_t col_00a = static_cast<uint16_t>(iter_m * 8u);
                    const uint16_t col_00b = static_cast<uint16_t>(col_00a + 2u);
                    const uint16_t col_01a = static_cast<uint16_t>(col_00a + 4u);
                    const uint16_t col_01b = static_cast<uint16_t>(col_00a + 6u);
                    LoadAlign(src_00a, src0_ub + col_00a * m);
                    LoadAlign(src_00b, src0_ub + col_00b * m);
                    LoadAlign(src_01a, src0_ub + col_01a * m);
                    LoadAlign(src_01b, src0_ub + col_01b * m);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, maskUb + col_00a * maskColStride);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, maskUb + col_00b * maskColStride);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, maskUb + col_01a * maskColStride);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, maskUb + col_01b * maskColStride);
                    Select(src_00a, src_00a, vreg_min, preg_compare0);
                    Select(src_00b, src_00b, vreg_min, preg_compare1);
                    Select(src_01a, src_01a, vreg_min, preg_compare2);
                    Select(src_01b, src_01b, vreg_min, preg_compare3);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src0_ub + col_00a * m, src_00a, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src0_ub + col_00b * m, src_00b, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src0_ub + col_01a * m, src_01a, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src0_ub + col_01b * m, src_01b, preg_108);
                }
                mem_bar(VST_VLD);
            }
        }
        for (uint16_t iter_m = 0; iter_m < maxLoopCount; ++iter_m) {
            const uint16_t col_00a = static_cast<uint16_t>(iter_m * 8u);
            const uint16_t col_00b = static_cast<uint16_t>(col_00a + 2u);
            const uint16_t col_01a = static_cast<uint16_t>(col_00a + 4u);
            const uint16_t col_01b = static_cast<uint16_t>(col_00a + 6u);
            LoadAlign(src_00a, src0_ub + col_00a * m);
            LoadAlign(src_00b, src0_ub + col_00b * m);
            LoadAlign(src_01a, src0_ub + col_01a * m);
            LoadAlign(src_01b, src0_ub + col_01b * m);
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, src_00a, preg_108);
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0b, max_0b, src_00b, preg_108);
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_1a, max_1a, src_01a, preg_108);
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_1b, max_1b, src_01b, preg_108);
        }
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, max_1a, preg_108);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0b, max_0b, max_1b, preg_108);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, max_0b, preg_108);
        vselr((MicroAPI::RegTensor<uint32_t>&)max_0b, (MicroAPI::RegTensor<uint32_t>&)max_0a,
              (MicroAPI::RegTensor<uint32_t>&)idx_swap);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, max_0b, preg_108);
        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, scale, preg_108);

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_max, max_0a, preg_all_b32);

        MicroAPI::Duplicate<T, T>(vreg_x_sum_even, 0);
        MicroAPI::Duplicate<T, T>(vreg_x_sum_odd, 0);
        uint16_t expLoopCount = static_cast<uint16_t>(ubN / packCols);

        static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t i0 = 0; i0 < expLoopCount; ++i0) {
            const uint16_t col_a = static_cast<uint16_t>(i0 * packCols);
            const uint16_t col_b = static_cast<uint16_t>(col_a + (packCols >> 1));
            LoadAlign(vreg_x_f32_a, ((__ubuf__ T*)input_x_local_UB) + col_a * m);
            LoadAlign(vreg_x_f32_b, ((__ubuf__ T*)input_x_local_UB) + col_b * m);
            if constexpr (hasAtten) {
                if (needAtten) {
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, maskUb + col_a * maskColStride);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, maskUb + col_b * maskColStride);
                    Select(vreg_x_f32_a, vreg_x_f32_a, vreg_min, preg_compare0);
                    Select(vreg_x_f32_b, vreg_x_f32_b, vreg_min, preg_compare1);
                }
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);

            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134);

            MicroAPI::Cast<T2, T, castTrait0>(vreg_x_exp_even_f16, vreg_x_exp_even, preg_135);
            MicroAPI::Cast<T2, T, castTrait0>(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_135);
            MicroAPI::DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2*&)x_exp, vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even,
                                                               preg_134);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd,
                                                               preg_134);
        }

        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134);
        vselr((MicroAPI::RegTensor<uint32_t>&)vreg_x_sum1, (MicroAPI::RegTensor<uint32_t>&)vreg_x_sum0,
              (MicroAPI::RegTensor<uint32_t>&)idx_swap);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(new_global_sum, vreg_x_sum0, preg_all_b32);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t s1BaseSize = 64, uint16_t ubN = 128,
          bool hasSink = false>
__aicore__ inline void ProcessVec1DnNoUpdate(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                             const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                             const LocalTensor<T>& expMaxTensor,
                                             const LocalTensor<uint8_t>& vselrIndexesBuf,
                                             const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n,
                                             const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                             float keepProb, bool needAtten, const float pScale = 1.0f,
                                             const float sinkValue = 0.0f)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t* maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = nullptr;
    float dScale = scale;

    if constexpr (s1BaseSize == 64) {
        ProcessVec1DnNoUpdate64VF<T, T2, hasAtten, s1BaseSize, ubN, hasSink>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, indexesUb, m, n, originN,
            scale, deScaleQK, minValue, keepProb, needAtten, dScale, pScale, sinkValue);
    } else if constexpr (s1BaseSize == 32) {
        ProcessVec1DnNoUpdate32VF<T, T2, hasAtten, s1BaseSize, ubN, hasSink>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, indexesUb, m, n, originN,
            scale, deScaleQK, minValue, keepProb, needAtten, dScale, pScale, sinkValue);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t s1BaseSize = 128, uint16_t ubN = 128,
          bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnUpdate64VF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                __ubuf__ float* new_global_max, __ubuf__ uint32_t* maskUb,
                                                __ubuf__ uint8_t* indexesUb, const uint32_t m, const uint32_t n,
                                                const uint32_t originN, const T scale, float deScaleQK,
                                                const T minValue, float keepProb, bool needAtten, const float dScale,
                                                const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    constexpr uint16_t repeatStride = 1;
    const uint16_t blockStride = (ubN >> 1) | 0x1;
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
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;

    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t* mask_ub0 = maskUb;
    __ubuf__ uint32_t* mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t* mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t* mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
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
    Muls(max0, max0, dScale, preg_108);
    if constexpr (hasSink) {
        Max(max0, max0, vreg_sink_input, preg_108);
    }

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
            Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
            Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
            Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }
        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1,
                         vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                             IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);

            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }
    RegTensor<float> vreg_l0;
    LoadAlign(vreg_l0, new_global_sum);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t s1BaseSize = 128, uint16_t ubN = 128,
          bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnUpdate32VF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                __ubuf__ float* new_global_max, __ubuf__ uint32_t* maskUb,
                                                __ubuf__ uint8_t* indexesUb, const uint32_t m, const uint32_t n,
                                                const uint32_t originN, const T scale, float deScaleQK,
                                                const T minValue, float keepProb, bool needAtten, const float dScale,
                                                const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    constexpr uint16_t repeatStride = 1;
    constexpr uint16_t packCols = 4u;
    const uint16_t blockStride = ubN / packCols | 0x1;
    const uint32_t maskColStride = m >> 2;
    __ubuf__ T* src0_ub = (__ubuf__ T*)input_x_local_UB;
    __ubuf__ T* src0_ub_1 = (__ubuf__ T*)input_x_local_UB + m;
    __ubuf__ T* src0_ub1 = src0_ub + m * 2;
    __ubuf__ T* src0_ub1_1 = src0_ub + m * 3;
    __ubuf__ T* src0_ub2 = src0_ub + m * 4;
    __ubuf__ T* src0_ub2_1 = src0_ub + m * 5;
    __ubuf__ T* src0_ub3 = src0_ub + m * 6;
    __ubuf__ T* src0_ub3_1 = src0_ub + m * 7;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vreg_x_sum_even;
        MicroAPI::RegTensor<T> vreg_x_sum_odd;
        MicroAPI::RegTensor<T> vreg_x_sum_1_even;
        MicroAPI::RegTensor<T> vreg_x_sum_1_odd;
        MicroAPI::RegTensor<T> vreg_x_sum0;
        MicroAPI::RegTensor<float> vreg_x_sum0_even;
        MicroAPI::RegTensor<float> vreg_x_sum0_odd;
        MicroAPI::RegTensor<T> vreg_x_sum1;
        MicroAPI::RegTensor<T2> vreg_x_exp_even_f16;
        MicroAPI::RegTensor<T2> vreg_x_exp_odd_f16;
        MicroAPI::RegTensor<T2> vreg_x_exp_even_bf16;
        MicroAPI::RegTensor<T2> vreg_x_exp_odd_bf16;

        MicroAPI::RegTensor<T> vreg_x_exp_even;
        MicroAPI::RegTensor<T> vreg_x_exp_odd;
        MicroAPI::RegTensor<T> vreg_x_f32_a;
        MicroAPI::RegTensor<T> vreg_x_f32_b;
        MicroAPI::RegTensor<T> vreg_x_exp_even_1;
        MicroAPI::RegTensor<T> vreg_x_exp_odd_1;
        MicroAPI::RegTensor<T2> vreg_x_exp_even_f16_1;
        MicroAPI::RegTensor<T2> vreg_x_exp_odd_f16_1;
        MicroAPI::RegTensor<T2> vreg_x_exp_even_bf16_1;
        MicroAPI::RegTensor<T2> vreg_x_exp_odd_bf16_1;

        MicroAPI::RegTensor<T> vreg_x_f32_1_a;
        MicroAPI::RegTensor<T> vreg_x_f32_1_b;
        MicroAPI::RegTensor<T2> vreg_x_exp_f16_pack;
        MicroAPI::RegTensor<T2> vreg_x_exp_f16_1_pack;
        MicroAPI::RegTensor<T2> vreg_x_exp_f16_packa;
        MicroAPI::RegTensor<T2> vreg_x_exp_f16_1_packa;
        MicroAPI::RegTensor<T2> vreg_x_exp_bf16_pack;
        MicroAPI::RegTensor<T2> vreg_x_exp_bf16_1_pack;
        MicroAPI::RegTensor<T2> vreg_x_exp_bf16_packa;
        MicroAPI::RegTensor<T2> vreg_x_exp_bf16_1_packa;
        MicroAPI::MaskReg preg_100;
        MicroAPI::MaskReg preg_101;
        MicroAPI::MaskReg preg_134;
        MicroAPI::MaskReg preg_135;
        MicroAPI::MaskReg preg_136;
        MicroAPI::MaskReg preg_all_b16;
        MicroAPI::MaskReg preg_all_b32;
        MicroAPI::MaskReg preg_all_s32;
        preg_134 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        preg_135 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        uint32_t sreg_92 = (uint32_t)128ULL;
        preg_136 = MicroAPI::UpdateMask<uint16_t>(sreg_92);
        preg_all_b16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        preg_all_b32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        preg_all_s32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_108;
        MicroAPI::MaskReg preg_compare0;
        MicroAPI::MaskReg preg_compare1;
        MicroAPI::MaskReg preg_compare2;
        MicroAPI::MaskReg preg_compare3;
        MicroAPI::RegTensor<int32_t> idx_swap;
        MicroAPI::RegTensor<int32_t> idx_swap16;
        MicroAPI::RegTensor<int32_t> idx_mask63;
        MicroAPI::RegTensor<T> src_00a, src_01a, src_02a, src_03a;
        MicroAPI::RegTensor<T> src_00b, src_01b, src_02b, src_03b;
        MicroAPI::RegTensor<T> src_10a, src_11a, src_12a, src_13a;
        MicroAPI::RegTensor<T> src_10b, src_11b, src_12b, src_13b;
        MicroAPI::RegTensor<T> max_0a, max_1a, max_2a, max_3a;
        MicroAPI::RegTensor<T> max_0b, max_1b, max_2b, max_3b;
        MicroAPI::RegTensor<T> vreg_min;
        MicroAPI::RegTensor<T> old_global_max;
        MicroAPI::RegTensor<T> old_global_sum;
        MicroAPI::RegTensor<T> vreg_exp_max;

        MicroAPI::Duplicate<T, T>(max_0a, minValue);
        MicroAPI::Duplicate<T, T>(max_0b, minValue);
        MicroAPI::Duplicate<T, T>(max_1a, minValue);
        MicroAPI::Duplicate<T, T>(max_1b, minValue);
        MicroAPI::Duplicate<T, T>(max_2a, minValue);
        MicroAPI::Duplicate<T, T>(max_2b, minValue);
        MicroAPI::Duplicate<T, T>(max_3a, minValue);
        MicroAPI::Duplicate<T, T>(max_3b, minValue);
        MicroAPI::Duplicate<T, T>(vreg_min, minValue);

        const uint16_t tailBegin = static_cast<uint16_t>(originN);
        bool fuseAlignedTail = false;

        preg_108 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange<int32_t>(idx_swap16, static_cast<int32_t>(16));
        MicroAPI::Arange<int32_t>(idx_swap, static_cast<int32_t>(32));
        MicroAPI::Duplicate<int32_t, int32_t>(idx_mask63, static_cast<int32_t>(63));
        MicroAPI::And<int32_t, MicroAPI::MaskMergeMode::ZEROING>(idx_swap16, idx_swap16, idx_mask63, preg_all_s32);
        MicroAPI::And<int32_t, MicroAPI::MaskMergeMode::ZEROING>(idx_swap, idx_swap, idx_mask63, preg_all_s32);
        uint16_t maxLoopCount = static_cast<uint16_t>(ubN / 8u);

        if constexpr (hasAtten) {
            if (needAtten) {
                for (uint16_t iter_m = 0; iter_m < maxLoopCount; ++iter_m) {
                    const uint16_t col_00a = static_cast<uint16_t>(iter_m * 8u);
                    const uint16_t col_00b = static_cast<uint16_t>(col_00a + 2u);
                    const uint16_t col_01a = static_cast<uint16_t>(col_00a + 4u);
                    const uint16_t col_01b = static_cast<uint16_t>(col_00a + 6u);
                    LoadAlign(src_00a, src0_ub + col_00a * m);
                    LoadAlign(src_00b, src0_ub + col_00b * m);
                    LoadAlign(src_01a, src0_ub + col_01a * m);
                    LoadAlign(src_01b, src0_ub + col_01b * m);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, maskUb + col_00a * maskColStride);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, maskUb + col_00b * maskColStride);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, maskUb + col_01a * maskColStride);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, maskUb + col_01b * maskColStride);
                    Select(src_00a, src_00a, vreg_min, preg_compare0);
                    Select(src_00b, src_00b, vreg_min, preg_compare1);
                    Select(src_01a, src_01a, vreg_min, preg_compare2);
                    Select(src_01b, src_01b, vreg_min, preg_compare3);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src0_ub + col_00a * m, src_00a, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src0_ub + col_00b * m, src_00b, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src0_ub + col_01a * m, src_01a, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src0_ub + col_01b * m, src_01b, preg_108);
                }
                mem_bar(VST_VLD);
            }
        }
        for (uint16_t iter_m = 0; iter_m < maxLoopCount; ++iter_m) {
            const uint16_t col_00a = static_cast<uint16_t>(iter_m * 8u);
            const uint16_t col_00b = static_cast<uint16_t>(col_00a + 2u);
            const uint16_t col_01a = static_cast<uint16_t>(col_00a + 4u);
            const uint16_t col_01b = static_cast<uint16_t>(col_00a + 6u);
            LoadAlign(src_00a, src0_ub + col_00a * m);
            LoadAlign(src_00b, src0_ub + col_00b * m);
            LoadAlign(src_01a, src0_ub + col_01a * m);
            LoadAlign(src_01b, src0_ub + col_01b * m);
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, src_00a, preg_108);
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0b, max_0b, src_00b, preg_108);
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_1a, max_1a, src_01a, preg_108);
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_1b, max_1b, src_01b, preg_108);
        }
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, max_1a, preg_108);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0b, max_0b, max_1b, preg_108);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, max_0b, preg_108);
        vselr((MicroAPI::RegTensor<uint32_t>&)max_0b, (MicroAPI::RegTensor<uint32_t>&)max_0a,
              (MicroAPI::RegTensor<uint32_t>&)idx_swap);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, max_0b, preg_108);
        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, scale, preg_108);

        LoadAlign(old_global_max, new_global_max);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(max_0a, max_0a, old_global_max, preg_108);
        MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_max, old_global_max, max_0a, preg_134);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)exp_max_fp32, vreg_exp_max, preg_all_b32);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_max, max_0a, preg_all_b32);

        MicroAPI::Duplicate<T, T>(vreg_x_sum_even, 0);
        MicroAPI::Duplicate<T, T>(vreg_x_sum_odd, 0);
        uint16_t expLoopCount = static_cast<uint16_t>(ubN / packCols);

        for (uint16_t i0 = 0; i0 < expLoopCount; ++i0) {
            const uint16_t col_a = static_cast<uint16_t>(i0 * packCols);
            const uint16_t col_b = static_cast<uint16_t>(col_a + (packCols >> 1));
            LoadAlign(vreg_x_f32_a, ((__ubuf__ T*)input_x_local_UB) + col_a * m);
            LoadAlign(vreg_x_f32_b, ((__ubuf__ T*)input_x_local_UB) + col_b * m);
            if constexpr (hasAtten) {
                if (needAtten) {
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, maskUb + col_a * maskColStride);
                    LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, maskUb + col_b * maskColStride);
                    Select(vreg_x_f32_a, vreg_x_f32_a, vreg_min, preg_compare0);
                    Select(vreg_x_f32_b, vreg_x_f32_b, vreg_min, preg_compare1);
                }
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);

            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134);

            static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            MicroAPI::Cast<T2, T, castTrait0>(vreg_x_exp_even_f16, vreg_x_exp_even, preg_135);
            MicroAPI::Cast<T2, T, castTrait0>(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_135);
            MicroAPI::DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2*&)x_exp, vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even,
                                                               preg_134);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd,
                                                               preg_134);
        }

        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134);
        vselr((MicroAPI::RegTensor<uint32_t>&)vreg_x_sum1, (MicroAPI::RegTensor<uint32_t>&)vreg_x_sum0,
              (MicroAPI::RegTensor<uint32_t>&)idx_swap);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
        LoadAlign(old_global_sum, new_global_sum);
        MicroAPI::Mul(old_global_sum, vreg_exp_max, old_global_sum, preg_134);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_x_sum0, vreg_x_sum0, old_global_sum, preg_134);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(new_global_sum, vreg_x_sum0, preg_all_b32);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t s1BaseSize = 128, uint16_t ubN = 128,
          bool hasSink = false>
__aicore__ inline void ProcessVec1DnUpdate(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                           const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                           const LocalTensor<T>& expMaxTensor,
                                           const LocalTensor<uint8_t>& vselrIndexesBuf,
                                           const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n,
                                           const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                           float keepProb, bool needAtten, const float pScale = 1.0f,
                                           const float sinkValue = 0.0f)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t* maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }
    if constexpr (s1BaseSize == 64) {
        ProcessVec1DnUpdate64VF<T, T2, hasAtten, s1BaseSize, ubN, hasSink>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, indexesUb, m, n, originN,
            scale, deScaleQK, minValue, keepProb, needAtten, dScale, pScale, sinkValue);
    } else if constexpr (s1BaseSize == 32) {
        ProcessVec1DnUpdate32VF<T, T2, hasAtten, s1BaseSize, ubN, hasSink>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, indexesUb, m, n, originN,
            scale, deScaleQK, minValue, keepProb, needAtten, dScale, pScale, sinkValue);
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

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t s1BaseSize = 128,
          uint16_t ubN = 256, bool hasSink = false>
__aicore__ inline void ProcessAntiquantVec1VfDn(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                                const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                                const LocalTensor<T>& expMaxTensor, TBuf<>* vselrIndexesBuf,
                                                const LocalTensor<uint8_t>& maskTensor, const uint32_t m,
                                                const uint32_t n, const uint32_t originN, const T scale,
                                                float deScaleQK, const T minValue, float keepProb, bool needAtten,
                                                const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    if constexpr (!isUpdate) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnNoUpdate<T, T2, hasAtten, s1BaseSize, ubN, hasSink>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, m, n, originN,
            scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnUpdate<T, T2, hasAtten, s1BaseSize, ubN, hasSink>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, m, n, originN,
            scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
    }
}

template <typename T>
__simd_vf__ inline void BroadCastMaxSumAntiquantVF(__ubuf__ float* out_ub, __ubuf__ float* ori_ub, const uint16_t loopM)
{
    RegTensor<float> broadcast_reg;
    MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
    for (uint16_t i = 0; i < loopM; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_E2B_B32>(broadcast_reg, ori_ub + i * 8);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)out_ub + i * 64, broadcast_reg, preg_all);
    }
}

template <typename T>
__aicore__ inline void BroadcastMaxSumAntiquant(const LocalTensor<T>& outTensor, const LocalTensor<T>& oriTensor,
                                                uint32_t vecS1RealSize)
{
    __ubuf__ float* out_ub = (__ubuf__ T*)outTensor.GetPhyAddr();
    __ubuf__ float* ori_ub = (__ubuf__ T*)oriTensor.GetPhyAddr();

    // Align8, broadcast one element to 8 elements, one register can store 64 elements,
    // so we can handle 64 / 8 = 8 elements per loop.
    uint16_t loopM = (vecS1RealSize + 7) >> 3;
    BroadCastMaxSumAntiquantVF<T>(out_ub, ori_ub, loopM);
}
} // namespace FaVectorApi
#endif // MUL_SEL_SOFTMAXFLASHV2_ANTIQUANT_CAST_NZ_DN_H_
