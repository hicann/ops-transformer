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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn_hif4.h
 * \brief hif4 dn softmax wrapper (ProcessVec1DnNoUpdateHif4, ProcessVec1DnUpdateHif4, ProcessVec1VfDnHif4)
 */

#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_HIF4_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_HIF4_H_

#include "kernel_tensor.h"
#include "../../arch35/vf/vf_basic_block_utils.h"

namespace FaVectorApi {
using namespace AscendC;
using namespace MicroAPI;
constexpr uint32_t BANK_STRIDE_B16 = 512;
constexpr uint32_t BANK_STRIDE_B8 = 1024;

constexpr static AscendC::MicroAPI::CastTrait castTraitTruncZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitTruncOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};
} // namespace FaVectorApi
namespace FaVectorApi {
using AscendC::LocalTensor;

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnHif4NoUpdateVF(__ubuf__ fp8_e4m3fn_t* x_exp, __ubuf__ T* input_x_local_UB,
                                                    __ubuf__ T* exp_max, __ubuf__ T* max, __ubuf__ T* global_max,
                                                    __ubuf__ uint8_t* indexesUb, __ubuf__ fp8_e8m0_t* pScaleSubLoop,
                                                    const uint32_t m, const uint32_t n, const uint32_t originN,
                                                    float pScale, const T minValue, int32_t subLoop, uint32_t c1v1Loop)
{
    RegTensor<half> vreg_x_exp_0;
    RegTensor<half> vreg_x_exp_1;
    RegTensor<half> vreg_x_exp_2;
    RegTensor<half> vreg_x_exp_3;
    RegTensor<half> vreg_x_exp_4;
    RegTensor<half> vreg_x_exp_5;
    RegTensor<half> vreg_x_exp_6;
    RegTensor<half> vreg_x_exp_7;

    RegTensor<half> vreg_x_fp16_0;
    RegTensor<half> vreg_x_fp16_1;
    RegTensor<half> vreg_x_fp16_2;
    RegTensor<half> vreg_x_fp16_3;
    RegTensor<half> vreg_x_fp16_4;
    RegTensor<half> vreg_x_fp16_5;
    RegTensor<half> vreg_x_fp16_6;
    RegTensor<half> vreg_x_fp16_7;

    RegTensor<half> vreg_x_max;
    RegTensor<half> vreg_exp_max_loop;
    RegTensor<half> vreg_x_max_subloop;
    RegTensor<half> vreg_expmax_subloop;
    RegTensor<half> vreg_min;

    RegTensor<half> vreg_p_scale;
    RegTensor<half> vreg_ln_p_scale;

    MaskReg preg_b8;
    MaskReg preg_b16;

    preg_b8 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    RegTensor<half> src0, src1, src2, src3;
    RegTensor<half> max0, max1, max2, max3, max_sub;

    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_0;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_1;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_2;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_3;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_4;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_5;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_6;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_7;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_f8_pack_0, vreg_x_exp_f8_pack_1;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_f8_pack_2, vreg_x_exp_f8_pack_3;

    RegTensor<uint16_t> vreg_x_exp_fp8_intlv_0, vreg_x_exp_fp8_intlv_1;
    RegTensor<uint16_t> vreg_x_exp_fp8_intlv_2, vreg_x_exp_fp8_intlv_3;
    RegTensor<uint32_t> vreg_x_exp_fp8_intlv_4, vreg_x_exp_fp8_intlv_5;
    RegTensor<uint32_t> vreg_x_exp_fp8_intlv_6, vreg_x_exp_fp8_intlv_7;
    RegTensor<uint8_t> idx_nd2nz;

    RegTensor<bfloat16_t> vreg_pscale_bf16_0, vreg_pscale_bf16_1;
    RegTensor<fp8_e8m0_t> vreg_pscale_f8e8m0_0, vreg_pscale_f8e8m0_1, vreg_pscale_f8e8m0_dst;

    __ubuf__ half* src_ub0 = input_x_local_UB; // 解bank冲突
    __ubuf__ half* src_ub1 = src_ub0 + BANK_STRIDE_B16;
    __ubuf__ half* src_ub2 = src_ub1 + BANK_STRIDE_B16;
    __ubuf__ half* src_ub3 = src_ub2 + BANK_STRIDE_B16;

    __ubuf__ fp8_e4m3fn_t* x_exp_0 = x_exp;
    __ubuf__ fp8_e4m3fn_t* x_exp_1 = x_exp_0 + 16 * BANK_STRIDE_B8; // NZ偏移
    __ubuf__ fp8_e4m3fn_t* x_exp_2 = x_exp_1 + 16 * BANK_STRIDE_B8;
    __ubuf__ fp8_e4m3fn_t* x_exp_3 = x_exp_2 + 16 * BANK_STRIDE_B8;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<half>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_b16);
    for (uint16_t i = originN; i < n; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)input_x_local_UB + i * BANK_STRIDE_B16,
                                                          vreg_min, preg_b16);
    }

    mem_bar(VST_VLD);

    for (uint16_t iter_m = 0; iter_m < uint16_t(n / 4); ++iter_m) {
        LoadAlign(src0, src_ub0 + iter_m * BANK_STRIDE_B16 * 4);
        LoadAlign(src1, src_ub1 + iter_m * BANK_STRIDE_B16 * 4);
        LoadAlign(src2, src_ub2 + iter_m * BANK_STRIDE_B16 * 4);
        LoadAlign(src3, src_ub3 + iter_m * BANK_STRIDE_B16 * 4);
        Max(max0, max0, src0, preg_b16);
        Max(max1, max1, src1, preg_b16);
        Max(max2, max2, src2, preg_b16);
        Max(max3, max3, src3, preg_b16);
    }

    Max(max0, max0, max2, preg_b16);
    Max(max1, max1, max3, preg_b16);
    Max(max0, max0, max1, preg_b16);
    Muls(max0, max0, INV_LN2, preg_b16);
    Truncate<half, RoundMode::CAST_CEIL>(max0, max0, preg_b16);
    Muls(max0, max0, LN2, preg_b16);

    if (subLoop != 0) {
        LoadAlign(vreg_x_max, max + (subLoop - 1) * m);
        Max(max0, max0, vreg_x_max, preg_b16);
    }

    if (subLoop == c1v1Loop) {
        StoreAlign<half, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ half*&)global_max, max0, preg_b16);
    }

    StoreAlign<half, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ half*&)max + subLoop * m, max0, preg_b16);
    Sub(max_sub, max0, vreg_ln_p_scale, preg_b16);
    LoadAlign(idx_nd2nz, indexesUb);
    for (uint16_t i0 = 0; i0 < n / 8; ++i0) { // 一次处理八行
        LoadAlign(vreg_x_fp16_0, input_x_local_UB + i0 * 8 * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_1, input_x_local_UB + (i0 * 8 + 1) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_2, input_x_local_UB + (i0 * 8 + 2) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_3, input_x_local_UB + (i0 * 8 + 3) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_4, input_x_local_UB + (i0 * 8 + 4) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_5, input_x_local_UB + (i0 * 8 + 5) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_6, input_x_local_UB + (i0 * 8 + 6) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_7, input_x_local_UB + (i0 * 8 + 7) * BANK_STRIDE_B16);

        FusedExpSub(vreg_x_exp_0, vreg_x_fp16_0, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_1, vreg_x_fp16_1, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_2, vreg_x_fp16_2, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_3, vreg_x_fp16_3, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_4, vreg_x_fp16_4, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_5, vreg_x_fp16_5, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_6, vreg_x_fp16_6, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_7, vreg_x_fp16_7, max_sub, preg_b16);

        Cast<fp8_e4m3fn_t, half, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintOne>(vreg_x_exp_fp8_1, vreg_x_exp_1, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintZero>(vreg_x_exp_fp8_2, vreg_x_exp_2, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintOne>(vreg_x_exp_fp8_3, vreg_x_exp_3, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintZero>(vreg_x_exp_fp8_4, vreg_x_exp_4, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintOne>(vreg_x_exp_fp8_5, vreg_x_exp_5, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintZero>(vreg_x_exp_fp8_6, vreg_x_exp_6, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintOne>(vreg_x_exp_fp8_7, vreg_x_exp_7, preg_b16);

        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_fp8_1, preg_b8);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_2,
           (RegTensor<uint8_t>&)vreg_x_exp_fp8_3, preg_b8);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_2, (RegTensor<uint8_t>&)vreg_x_exp_fp8_4,
           (RegTensor<uint8_t>&)vreg_x_exp_fp8_5, preg_b8);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_3, (RegTensor<uint8_t>&)vreg_x_exp_fp8_6,
           (RegTensor<uint8_t>&)vreg_x_exp_fp8_7, preg_b8);

        Interleave((RegTensor<uint16_t>&)vreg_x_exp_fp8_intlv_0, (RegTensor<uint16_t>&)vreg_x_exp_fp8_intlv_1,
                   (RegTensor<uint16_t>&)vreg_x_exp_fp8_0, (RegTensor<uint16_t>&)vreg_x_exp_fp8_1);
        Interleave((RegTensor<uint16_t>&)vreg_x_exp_fp8_intlv_2, (RegTensor<uint16_t>&)vreg_x_exp_fp8_intlv_3,
                   (RegTensor<uint16_t>&)vreg_x_exp_fp8_2, (RegTensor<uint16_t>&)vreg_x_exp_fp8_3);

        Interleave((RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_4, (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_5,
                   (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_0, (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_2);
        Interleave((RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_6, (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_7,
                   (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_1, (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_3);

        Gather((RegTensor<uint8_t>&)vreg_x_exp_f8_pack_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_intlv_4, idx_nd2nz);
        Gather((RegTensor<uint8_t>&)vreg_x_exp_f8_pack_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_intlv_5, idx_nd2nz);
        Gather((RegTensor<uint8_t>&)vreg_x_exp_f8_pack_2, (RegTensor<uint8_t>&)vreg_x_exp_fp8_intlv_6, idx_nd2nz);
        Gather((RegTensor<uint8_t>&)vreg_x_exp_f8_pack_3, (RegTensor<uint8_t>&)vreg_x_exp_fp8_intlv_7, idx_nd2nz);

        StoreAlign(((__ubuf__ uint8_t*&)x_exp_0) + i0 * BANK_STRIDE_B8, (RegTensor<uint8_t>&)vreg_x_exp_f8_pack_0,
                   preg_b8);
        StoreAlign(((__ubuf__ uint8_t*&)x_exp_1) + i0 * BANK_STRIDE_B8, (RegTensor<uint8_t>&)vreg_x_exp_f8_pack_1,
                   preg_b8);
        StoreAlign(((__ubuf__ uint8_t*&)x_exp_2) + i0 * BANK_STRIDE_B8, (RegTensor<uint8_t>&)vreg_x_exp_f8_pack_2,
                   preg_b8);
        StoreAlign(((__ubuf__ uint8_t*&)x_exp_3) + i0 * BANK_STRIDE_B8, (RegTensor<uint8_t>&)vreg_x_exp_f8_pack_3,
                   preg_b8);
    }

    if (subLoop == c1v1Loop) {
        for (uint16_t i = 0; i <= c1v1Loop; i++) {
            LoadAlign(vreg_x_max_subloop, max + i * m);
            FusedExpSub(vreg_expmax_subloop, vreg_x_max_subloop, max0, preg_b16);

            Cast<bfloat16_t, half, castTraitRintZero>(vreg_pscale_bf16_0, vreg_expmax_subloop, preg_b16);
            Cast<fp8_e8m0_t, bfloat16_t, castTraitTruncZero>(vreg_pscale_f8e8m0_0, vreg_pscale_bf16_0, preg_b16);
            Cast<fp8_e8m0_t, bfloat16_t, castTraitTruncOne>(vreg_pscale_f8e8m0_1, vreg_pscale_bf16_0, preg_b16);
            Or((RegTensor<uint8_t>&)vreg_pscale_f8e8m0_dst, (RegTensor<uint8_t>&)vreg_pscale_f8e8m0_0,
               (RegTensor<uint8_t>&)vreg_pscale_f8e8m0_1, preg_b8);
            StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t*&)pScaleSubLoop + i * 256),
                                                                      vreg_pscale_f8e8m0_dst, preg_b8);
        }
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnHif4UpdateVF(__ubuf__ fp8_e4m3fn_t* x_exp, __ubuf__ T* input_x_local_UB,
                                                  __ubuf__ T* exp_max, __ubuf__ T* max, __ubuf__ T* global_max,
                                                  __ubuf__ uint8_t* indexesUb, __ubuf__ fp8_e8m0_t* pScaleSubLoop,
                                                  const uint32_t m, const uint32_t n, const uint32_t originN,
                                                  float pScale, const T minValue, int32_t subLoop, uint32_t c1v1Loop)
{
    RegTensor<half> vreg_x_exp_0;
    RegTensor<half> vreg_x_exp_1;
    RegTensor<half> vreg_x_exp_2;
    RegTensor<half> vreg_x_exp_3;
    RegTensor<half> vreg_x_exp_4;
    RegTensor<half> vreg_x_exp_5;
    RegTensor<half> vreg_x_exp_6;
    RegTensor<half> vreg_x_exp_7;

    RegTensor<half> vreg_x_fp16_0;
    RegTensor<half> vreg_x_fp16_1;
    RegTensor<half> vreg_x_fp16_2;
    RegTensor<half> vreg_x_fp16_3;
    RegTensor<half> vreg_x_fp16_4;
    RegTensor<half> vreg_x_fp16_5;
    RegTensor<half> vreg_x_fp16_6;
    RegTensor<half> vreg_x_fp16_7;

    RegTensor<half> vreg_x_max;
    RegTensor<half> vreg_min;
    RegTensor<half> vreg_exp_max_loop;
    RegTensor<half> vreg_x_max_subloop;
    RegTensor<half> vreg_expmax_subloop;

    RegTensor<half> vreg_p_scale;
    RegTensor<half> vreg_ln_p_scale;

    MaskReg preg_b8;
    MaskReg preg_b16;

    preg_b8 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    RegTensor<half> src0, src1, src2, src3;
    RegTensor<half> max0, max1, max2, max3, max_sub;

    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_0;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_1;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_2;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_3;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_4;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_5;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_6;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_fp8_7;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_f8_pack_0, vreg_x_exp_f8_pack_1;
    RegTensor<fp8_e4m3fn_t> vreg_x_exp_f8_pack_2, vreg_x_exp_f8_pack_3;
    RegTensor<uint16_t> vreg_x_exp_fp8_intlv_0, vreg_x_exp_fp8_intlv_1;
    RegTensor<uint16_t> vreg_x_exp_fp8_intlv_2, vreg_x_exp_fp8_intlv_3;
    RegTensor<uint32_t> vreg_x_exp_fp8_intlv_4, vreg_x_exp_fp8_intlv_5;
    RegTensor<uint32_t> vreg_x_exp_fp8_intlv_6, vreg_x_exp_fp8_intlv_7;
    RegTensor<uint8_t> idx_nd2nz;

    RegTensor<bfloat16_t> vreg_pscale_bf16_0, vreg_pscale_bf16_1;
    RegTensor<fp8_e8m0_t> vreg_pscale_f8e8m0_0, vreg_pscale_f8e8m0_1, vreg_pscale_f8e8m0_dst;

    __ubuf__ half* src_ub0 = input_x_local_UB;
    __ubuf__ half* src_ub1 = src_ub0 + BANK_STRIDE_B16;
    __ubuf__ half* src_ub2 = src_ub1 + BANK_STRIDE_B16;
    __ubuf__ half* src_ub3 = src_ub2 + BANK_STRIDE_B16;

    __ubuf__ fp8_e4m3fn_t* x_exp_0 = x_exp;
    __ubuf__ fp8_e4m3fn_t* x_exp_1 = x_exp_0 + 16 * BANK_STRIDE_B8; //
    __ubuf__ fp8_e4m3fn_t* x_exp_2 = x_exp_1 + 16 * BANK_STRIDE_B8;
    __ubuf__ fp8_e4m3fn_t* x_exp_3 = x_exp_2 + 16 * BANK_STRIDE_B8;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<half>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_b16);
    for (uint16_t i = originN; i < n; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)input_x_local_UB + i * BANK_STRIDE_B16,
                                                          vreg_min, preg_b16);
    }

    mem_bar(VST_VLD);

    for (uint16_t iter_m = 0; iter_m < uint16_t(n / 4); ++iter_m) {
        LoadAlign(src0, src_ub0 + iter_m * BANK_STRIDE_B16 * 4);
        LoadAlign(src1, src_ub1 + iter_m * BANK_STRIDE_B16 * 4);
        LoadAlign(src2, src_ub2 + iter_m * BANK_STRIDE_B16 * 4);
        LoadAlign(src3, src_ub3 + iter_m * BANK_STRIDE_B16 * 4);
        Max(max0, max0, src0, preg_b16);
        Max(max1, max1, src1, preg_b16);
        Max(max2, max2, src2, preg_b16);
        Max(max3, max3, src3, preg_b16);
    }

    Max(max0, max0, max2, preg_b16);
    Max(max1, max1, max3, preg_b16);
    Max(max0, max0, max1, preg_b16);
    Muls(max0, max0, INV_LN2, preg_b16);
    Truncate<half, RoundMode::CAST_CEIL>(max0, max0, preg_b16);
    Muls(max0, max0, LN2, preg_b16);

    if (subLoop == 0) {
        LoadAlign(vreg_x_max, global_max);
        Max(max0, max0, vreg_x_max, preg_b16);
    } else {
        LoadAlign(vreg_x_max, max + (subLoop - 1) * m);
        Max(max0, max0, vreg_x_max, preg_b16);
    }
    if (subLoop == c1v1Loop) {
        LoadAlign(vreg_x_max, global_max);
        FusedExpSub(vreg_exp_max_loop, vreg_x_max, max0, preg_b16);
        StoreAlign<half, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ half*&)global_max, max0, preg_b16);
        StoreAlign<half, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ half*&)exp_max, vreg_exp_max_loop, preg_b16);
    }
    StoreAlign<half, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ half*&)max + subLoop * m, max0, preg_b16);
    Sub(max_sub, max0, vreg_ln_p_scale, preg_b16);
    LoadAlign(idx_nd2nz, indexesUb);
    for (uint16_t i0 = 0; i0 < n / 8; ++i0) {
        LoadAlign(vreg_x_fp16_0, input_x_local_UB + i0 * 8 * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_1, input_x_local_UB + (i0 * 8 + 1) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_2, input_x_local_UB + (i0 * 8 + 2) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_3, input_x_local_UB + (i0 * 8 + 3) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_4, input_x_local_UB + (i0 * 8 + 4) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_5, input_x_local_UB + (i0 * 8 + 5) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_6, input_x_local_UB + (i0 * 8 + 6) * BANK_STRIDE_B16);
        LoadAlign(vreg_x_fp16_7, input_x_local_UB + (i0 * 8 + 7) * BANK_STRIDE_B16);

        FusedExpSub(vreg_x_exp_0, vreg_x_fp16_0, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_1, vreg_x_fp16_1, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_2, vreg_x_fp16_2, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_3, vreg_x_fp16_3, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_4, vreg_x_fp16_4, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_5, vreg_x_fp16_5, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_6, vreg_x_fp16_6, max_sub, preg_b16);
        FusedExpSub(vreg_x_exp_7, vreg_x_fp16_7, max_sub, preg_b16);

        Cast<fp8_e4m3fn_t, half, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintOne>(vreg_x_exp_fp8_1, vreg_x_exp_1, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintZero>(vreg_x_exp_fp8_2, vreg_x_exp_2, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintOne>(vreg_x_exp_fp8_3, vreg_x_exp_3, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintZero>(vreg_x_exp_fp8_4, vreg_x_exp_4, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintOne>(vreg_x_exp_fp8_5, vreg_x_exp_5, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintZero>(vreg_x_exp_fp8_6, vreg_x_exp_6, preg_b16);
        Cast<fp8_e4m3fn_t, half, castTraitRintOne>(vreg_x_exp_fp8_7, vreg_x_exp_7, preg_b16);

        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_fp8_1, preg_b8);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_2,
           (RegTensor<uint8_t>&)vreg_x_exp_fp8_3, preg_b8);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_2, (RegTensor<uint8_t>&)vreg_x_exp_fp8_4,
           (RegTensor<uint8_t>&)vreg_x_exp_fp8_5, preg_b8);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_3, (RegTensor<uint8_t>&)vreg_x_exp_fp8_6,
           (RegTensor<uint8_t>&)vreg_x_exp_fp8_7, preg_b8);

        Interleave((RegTensor<uint16_t>&)vreg_x_exp_fp8_intlv_0, (RegTensor<uint16_t>&)vreg_x_exp_fp8_intlv_1,
                   (RegTensor<uint16_t>&)vreg_x_exp_fp8_0, (RegTensor<uint16_t>&)vreg_x_exp_fp8_1);
        Interleave((RegTensor<uint16_t>&)vreg_x_exp_fp8_intlv_2, (RegTensor<uint16_t>&)vreg_x_exp_fp8_intlv_3,
                   (RegTensor<uint16_t>&)vreg_x_exp_fp8_2, (RegTensor<uint16_t>&)vreg_x_exp_fp8_3);

        Interleave((RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_4, (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_5,
                   (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_0, (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_2);
        Interleave((RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_6, (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_7,
                   (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_1, (RegTensor<uint32_t>&)vreg_x_exp_fp8_intlv_3);

        Gather((RegTensor<uint8_t>&)vreg_x_exp_f8_pack_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_intlv_4, idx_nd2nz);
        Gather((RegTensor<uint8_t>&)vreg_x_exp_f8_pack_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_intlv_5, idx_nd2nz);
        Gather((RegTensor<uint8_t>&)vreg_x_exp_f8_pack_2, (RegTensor<uint8_t>&)vreg_x_exp_fp8_intlv_6, idx_nd2nz);
        Gather((RegTensor<uint8_t>&)vreg_x_exp_f8_pack_3, (RegTensor<uint8_t>&)vreg_x_exp_fp8_intlv_7, idx_nd2nz);

        StoreAlign(((__ubuf__ uint8_t*&)x_exp_0) + i0 * BANK_STRIDE_B8, (RegTensor<uint8_t>&)vreg_x_exp_f8_pack_0,
                   preg_b8);
        StoreAlign(((__ubuf__ uint8_t*&)x_exp_1) + i0 * BANK_STRIDE_B8, (RegTensor<uint8_t>&)vreg_x_exp_f8_pack_1,
                   preg_b8);
        StoreAlign(((__ubuf__ uint8_t*&)x_exp_2) + i0 * BANK_STRIDE_B8, (RegTensor<uint8_t>&)vreg_x_exp_f8_pack_2,
                   preg_b8);
        StoreAlign(((__ubuf__ uint8_t*&)x_exp_3) + i0 * BANK_STRIDE_B8, (RegTensor<uint8_t>&)vreg_x_exp_f8_pack_3,
                   preg_b8);
    }

    if (subLoop == c1v1Loop) {
        for (uint16_t i = 0; i <= c1v1Loop; i++) {
            LoadAlign(vreg_x_max_subloop, max + i * m);
            FusedExpSub(vreg_expmax_subloop, vreg_x_max_subloop, max0, preg_b16);

            Cast<bfloat16_t, half, castTraitRintZero>(vreg_pscale_bf16_0, vreg_expmax_subloop, preg_b16);
            Cast<fp8_e8m0_t, bfloat16_t, castTraitTruncZero>(vreg_pscale_f8e8m0_0, vreg_pscale_bf16_0, preg_b16);
            Cast<fp8_e8m0_t, bfloat16_t, castTraitTruncOne>(vreg_pscale_f8e8m0_1, vreg_pscale_bf16_0, preg_b16);
            Or((RegTensor<uint8_t>&)vreg_pscale_f8e8m0_dst, (RegTensor<uint8_t>&)vreg_pscale_f8e8m0_0,
               (RegTensor<uint8_t>&)vreg_pscale_f8e8m0_1, preg_b8);
            StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t*&)pScaleSubLoop + i * 256),
                                                                      vreg_pscale_f8e8m0_dst, preg_b8);
        }
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnNoUpdateHif4(const uint64_t& dstTensor, const uint64_t& maxTensor,
                                                 const uint64_t& globalMaxTensor, const LocalTensor<T> srcTensor,
                                                 const uint64_t& expMaxTensor,
                                                 const LocalTensor<uint8_t>& vselrIndexesBuf, const uint32_t m,
                                                 const uint32_t n, const uint32_t originN, float pScale,
                                                 const T minValue, int32_t subLoop, uint32_t c1v1Loop,
                                                 const uint64_t& pScaleSubLoopTensor)
{
    __ubuf__ fp8_e4m3fn_t* x_exp = (__ubuf__ fp8_e4m3fn_t*)dstTensor;
    __ubuf__ half* input_x_local_UB = (__ubuf__ half*)srcTensor.GetPhyAddr();
    __ubuf__ half* exp_max = (__ubuf__ half*)expMaxTensor;
    __ubuf__ half* max = (__ubuf__ half*)maxTensor;
    __ubuf__ half* global_max = (__ubuf__ half*)globalMaxTensor;
    __ubuf__ fp8_e8m0_t* pScaleSubLoopUb = (__ubuf__ fp8_e8m0_t*)pScaleSubLoopTensor;
    __ubuf__ uint8_t* indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();

    ProcessVec1DnHif4NoUpdateVF<T, T2, hasAtten, ubN>(x_exp, input_x_local_UB, exp_max, max, global_max, indexesUb,
                                                      pScaleSubLoopUb, m, n, originN, pScale, minValue, subLoop,
                                                      c1v1Loop);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnUpdateHif4(const uint64_t& dstTensor, const uint64_t& maxTensor,
                                               const uint64_t& globalMaxTensor, const LocalTensor<T> srcTensor,
                                               const uint64_t& expMaxTensor,
                                               const LocalTensor<uint8_t>& vselrIndexesBuf, const uint32_t m,
                                               const uint32_t n, const uint32_t originN, float pScale, const T minValue,
                                               int32_t subLoop, uint32_t c1v1Loop, const uint64_t& pScaleSubLoopTensor)
{
    __ubuf__ fp8_e4m3fn_t* x_exp = (__ubuf__ fp8_e4m3fn_t*)dstTensor;
    __ubuf__ half* input_x_local_UB = (__ubuf__ half*)srcTensor.GetPhyAddr();
    __ubuf__ half* exp_max = (__ubuf__ half*)expMaxTensor;
    __ubuf__ half* max = (__ubuf__ half*)maxTensor;
    __ubuf__ half* global_max = (__ubuf__ half*)globalMaxTensor;
    __ubuf__ fp8_e8m0_t* pScaleSubLoopUb = (__ubuf__ fp8_e8m0_t*)pScaleSubLoopTensor;
    __ubuf__ uint8_t* indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();

    ProcessVec1DnHif4UpdateVF<T, T2, hasAtten, ubN>(x_exp, input_x_local_UB, exp_max, max, global_max, indexesUb,
                                                    pScaleSubLoopUb, m, n, originN, pScale, minValue, subLoop,
                                                    c1v1Loop);
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1VfDnHif4(const uint64_t& dstTensor, const uint64_t& maxTensor,
                                           const uint64_t& globalMaxTensor, const LocalTensor<T> srcTensor,
                                           const uint64_t& expMaxTensor, const LocalTensor<uint8_t> indexesTensor,
                                           const uint64_t& pScaleSubLoopTensor, const uint32_t m, const uint32_t n,
                                           const uint32_t originN, float pScale, const T minValue, int32_t subLoop,
                                           uint32_t c1v1Loop)
{
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdateHif4<T, T2, hasAtten, ubN>(dstTensor, maxTensor, globalMaxTensor, srcTensor, expMaxTensor,
                                                        indexesTensor, m, n, originN, pScale, minValue, subLoop,
                                                        c1v1Loop, pScaleSubLoopTensor);
    } else {
        ProcessVec1DnUpdateHif4<T, T2, hasAtten, ubN>(dstTensor, maxTensor, globalMaxTensor, srcTensor, expMaxTensor,
                                                      indexesTensor, m, n, originN, pScale, minValue, subLoop, c1v1Loop,
                                                      pScaleSubLoopTensor);
    }
}

} // namespace FaVectorApi

#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_HIF4_H_
