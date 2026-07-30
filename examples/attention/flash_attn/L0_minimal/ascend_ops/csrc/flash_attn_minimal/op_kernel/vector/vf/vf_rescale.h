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
 * \file vf_rescale.h
 * \brief
 */
#ifndef VF_RESCALE_H
#define VF_RESCALE_H

#include "kernel_tensor.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#ifdef __NPU_DEVICE__
namespace FaVectorApi {
#ifndef __CCE_KT_TEST__
// bf16->fp32
static constexpr MicroAPI::CastTrait castTraitFp16_32_update = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr uint16_t REDUCE_SIZE = 1;
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre>
__simd_vf__ inline void FlashUpdateBasicVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * rowMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);

        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_add, preg_all);
        }
    }
}
/* **************************************************************************************************
 * FlashUpdate, fp32
 * ************************************************************************************************* */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre>
__aicore__ inline void FlashUpdateBasic(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor,
    const uint16_t m, const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * rowMaxUb = (__ubuf__ T*)rowMaxTensor.GetPhyAddr();

    FlashUpdateBasicVF<T, INPUT_T, OUTPUT_T, srcD, reduceSize, isUpdatePre>(
        dstUb, curUb, preUb, expMaxUb, rowMaxUb, m, d, deScaleV, deScaleVPre);
}

/*
 * @ingroup FlashUpdate
 * @brief compute, dstTensor = preTensor * expMaxTensor + curTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, bool isUpdatePre>
__aicore__ inline void FlashUpdate(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdate, T must be float");
    FlashUpdateBasic<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE, isUpdatePre>(dstTensor, curTensor, preTensor, expMaxTensor, rowMaxTensor,
        m, d, deScaleV, deScaleVPre);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre>
__simd_vf__ inline void FlashUpdateLastBasicVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * rowMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_all);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre>
__aicore__ inline void FlashUpdateLastBasic(const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& curTensor, const LocalTensor<T>& preTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m, const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * rowMaxUb = (__ubuf__ T*)rowMaxTensor.GetPhyAddr();

    FlashUpdateLastBasicVF<T, INPUT_T, OUTPUT_T, srcD, reduceSize, isUpdatePre>(
        dstUb, curUb, preUb, expMaxUb, expSumUb, rowMaxUb, m, d, deScaleV, deScaleVPre);
}

/*
 * @ingroup FlashUpdateLast
 * @brief compute, dstTensor = (preTensor * expMaxTensor + curTensor) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, 32 bytes align
 */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, bool isUpdatePre>
__aicore__ inline void FlashUpdateLast(const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& curTensor, const LocalTensor<T>& preTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor, const LocalTensor<T>& expSumTensor,
    uint16_t m, uint16_t d, const float deScaleV, const float deScaleVPre)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateLast, T must be float");
    FlashUpdateLastBasic<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE, isUpdatePre>(
        dstTensor, curTensor, preTensor, expMaxTensor, rowMaxTensor, expSumTensor, m, d, deScaleV, deScaleVPre);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD>
__simd_vf__ inline void LastDivVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * expSumUb,
    const uint16_t m, const uint16_t d, const float deScaleV)
{
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    const uint16_t dLoops = d >> 6;

    for (uint16_t i = 0; i < m; ++i) {
        uint32_t sreg_init = d;
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * REDUCE_SIZE);
        for (uint16_t j = 0; j < dLoops; ++j) {
            MaskReg preg_update = UpdateMask<float>(sreg_init);

            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            Div(vreg_div, vreg_input_cur, vreg_exp_sum, preg_update);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_update);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD>
__aicore__ inline void LastDiv(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d, const float deScaleV)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    LastDivVF<T, INPUT_T, OUTPUT_T, srcD>(dstUb, curUb, expSumUb, m, d, deScaleV);
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__simd_vf__ inline void DivCastImpl128VF(__ubuf__ OUTPUT_T * dstUb, __ubuf__ float * srcUb, __ubuf__ float * expSumUb,
    const uint16_t m)
{
    RegTensor<float> vreg_src_even, vreg_src_odd;
    RegTensor<float> vreg_div_even, vreg_div_odd;
    RegTensor<float> vreg_exp_sum;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_div_even_bf16;
    RegTensor<bfloat16_t> vreg_div_odd_bf16;
    RegTensor<bfloat16_t> vreg_cast_bf16;
    // half
    RegTensor<half> vreg_div_even_f16;
    RegTensor<half> vreg_div_odd_f16;
    RegTensor<half> vreg_cast_f16;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i);
        if constexpr (IsSameType<OUTPUT_T, float>::value) {
            LoadAlign(vreg_src_even, srcUb + i * srcD);
            LoadAlign(vreg_src_odd, srcUb + i * srcD + (srcD >> 1));
        } else {
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_src_even, vreg_src_odd, srcUb + i * srcD);
        }
        Div(vreg_div_even, vreg_src_even, vreg_exp_sum, preg_all);
        Div(vreg_div_odd, vreg_src_odd, vreg_exp_sum, preg_all);

        if constexpr (IsSameType<OUTPUT_T, float>::value) {
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_div_even, preg_all);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD + (srcD >> 1), vreg_div_odd, preg_all);
        } else if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_even_bf16, vreg_div_even, preg_all);
            Cast<OUTPUT_T, T, castTraitOne>(vreg_div_odd_bf16, vreg_div_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_cast_bf16, (RegTensor<uint16_t>&)vreg_div_even_bf16,
                (RegTensor<uint16_t>&)vreg_div_odd_bf16, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_cast_bf16, preg_all_b16);
        } else {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_even_f16, vreg_div_even, preg_all);
            Cast<OUTPUT_T, T, castTraitOne>(vreg_div_odd_f16, vreg_div_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_cast_f16, (RegTensor<uint16_t>&)vreg_div_even_f16, (RegTensor<uint16_t>&)vreg_div_odd_f16, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_cast_f16, preg_all_b16);
        }
    }
}

/*
 * @ingroup DivCast
 * @brief compute, dstTensor = cast(srcTensor / expSumTensor)
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor, shape is [m, 8]
 * @param [in] m, input rows
 * @param [in] d, input colums, 32 bytes align
 * @param [in] srcD, should be 64 or 128
 */
template <typename T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void DivCast(const LocalTensor<OUTPUT_T>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m)
{
    __ubuf__ OUTPUT_T * dstUb = (__ubuf__ OUTPUT_T*)dstTensor.GetPhyAddr();
    __ubuf__ float * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    DivCastImpl128VF<T, OUTPUT_T, srcD>(dstUb, srcUb, expSumUb, m);
}
#endif
} // namespace
#endif
#endif
