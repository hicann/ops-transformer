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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_regbase_v2.h
 * \brief
 */
#ifndef MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_REGBASE_V2_INTERFACE_H
#define MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_REGBASE_V2_INTERFACE_H

#include "kernel_tensor.h"
#include "../pse_arch38.h"

using namespace regbaseutil;

namespace FaVectorApi {
/* **************************************************************************************************
 * only 128*128 support
 * only high performance support (expSum expMax use
 * fp32)************************************************************************************************* */
// originN = 128, No update
constexpr uint32_t floatRepSize = 64;
constexpr uint32_t halfRepSize = 128;
constexpr uint32_t blockBytesU8 = 32;

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510NoUpdateImpl128(
    const LocalTensor<T2> &dstTensor, const LocalTensor<float> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<float> &expMaxTensor, const LocalTensor<T> &inSrcTensor, const LocalTensor<float> &inExpSumTensor,
    const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &inMaskTensor, const LocalTensor<T> &inPseTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 *expUb = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ float *expSumUb = (__ubuf__ float *)expSumTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *maxUbStart = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)inSrcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vreg_input_x;
        Reg::RegTensor<T> vreg_input_max;
        Reg::RegTensor<T> vreg_max_brc;
        Reg::RegTensor<float> vreg_exp_sum;
        Reg::RegTensor<float> vreg_exp_even;
        Reg::RegTensor<float> vreg_exp_odd;

        Reg::UnalignReg ureg_max;
        Reg::UnalignReg ureg_exp_sum;

        Reg::MaskReg preg_all_b32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_all_b16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_all_b8 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_s8 = Reg::CreateMask<int8_t, Reg::MaskPattern::VL128>();

        Reg::RegTensor<half> vreg_exp_res;
        Reg::RegTensor<half> vreg_muls_res;
        Reg::RegTensor<T2> vreg_cast;
        Reg::RegTensor<T2> vreg_res;

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x,
                                                       srcUb + i * sInner); // fp16 data 256B one row
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_input_x, vreg_input_x, scale,
                                                         preg_all_b16); // Muls(scale)
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner, vreg_input_x, preg_all_b16);
            Reg::ReduceMax<T, Reg::MaskMergeMode::ZEROING>(vreg_input_max, vreg_input_x, preg_all_b16);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(maxUb, vreg_input_max, ureg_max, 1);
        }
        Reg::DataCopyUnAlignPost<T, Reg::PostLiteral::POST_MODE_UPDATE>(maxUb, ureg_max, 0);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B16>(vreg_max_brc, maxUbStart + i);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x, srcUb + i * sInner);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_exp_res, vreg_input_x,
                                                                                     vreg_max_brc, preg_all_b16);

            static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2 *&)expUb, vreg_exp_res, blockStride, repeatStride, preg_all_b16);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Reg::Cast<float, half, castTrait0>(vreg_exp_even, vreg_exp_res, preg_all_b16);
            Reg::Cast<float, half, castTrait1>(vreg_exp_odd, vreg_exp_res, preg_all_b16);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all_b32);
            Reg::ReduceSum<float, float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_sum, preg_all_b32);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(expSumUb, vreg_exp_sum, ureg_exp_sum, 1);
        }
        Reg::DataCopyUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(expSumUb, ureg_exp_sum, 0);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510NoUpdateImpl256(
    const LocalTensor<T2> &dstTensor, const LocalTensor<float> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<float> &expMaxTensor, const LocalTensor<T> &inSrcTensor, const LocalTensor<float> &inExpSumTensor,
    const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &inMaskTensor, const LocalTensor<T> &inPseTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 *expUb1 = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T2 *expUb2 = (__ubuf__ T2 *)dstTensor.GetPhyAddr() + blockStride * 128;
    __ubuf__ float *expSumUb = (__ubuf__ float *)expSumTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *maxUbStart = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)inSrcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vreg_input_x_1;
        Reg::RegTensor<T> vreg_input_x_2;
        Reg::RegTensor<T> vreg_input_max_tmp;
        Reg::RegTensor<T> vreg_input_max;
        Reg::RegTensor<T> vreg_max_brc;
        Reg::RegTensor<float> vreg_exp_sum;
        Reg::RegTensor<float> vreg_exp_even;
        Reg::RegTensor<float> vreg_exp_odd;

        Reg::UnalignReg ureg_max;
        Reg::UnalignReg ureg_exp_sum;

        Reg::MaskReg preg_all_b32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_all_b16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_all_b8 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_s8 = Reg::CreateMask<int8_t, Reg::MaskPattern::VL128>();

        Reg::RegTensor<T> vreg_exp_res;
        Reg::RegTensor<T> vreg_exp_res_1;
        Reg::RegTensor<T> vreg_exp_res_2;

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x_1, srcUb + i * sInner);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb + i * sInner + halfRepSize);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_input_x_1, vreg_input_x_1, scale, preg_all_b16);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_input_x_2, vreg_input_x_2, scale, preg_all_b16);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner, vreg_input_x_1, preg_all_b16);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner + halfRepSize, vreg_input_x_2,
                                                            preg_all_b16);
            Reg::Max(vreg_input_max_tmp, vreg_input_x_1, vreg_input_x_2, preg_all_b16);
            Reg::ReduceMax<T, Reg::MaskMergeMode::ZEROING>(vreg_input_max, vreg_input_max_tmp, preg_all_b16);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(maxUb, vreg_input_max, ureg_max, 1);
        }
        Reg::DataCopyUnAlignPost<T, Reg::PostLiteral::POST_MODE_UPDATE>(maxUb, ureg_max, 0);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B16>(vreg_max_brc, maxUbStart + i);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x_1, srcUb + i * sInner);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb + i * sInner + halfRepSize);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_exp_res_1, vreg_input_x_1,
                                                                                     vreg_max_brc, preg_all_b16);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_exp_res_2, vreg_input_x_2,
                                                                                     vreg_max_brc, preg_all_b16);
            static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

            Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2 *&)expUb1, vreg_exp_res_1, blockStride, repeatStride, preg_all_b16);
            Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2 *&)expUb2, vreg_exp_res_2, blockStride, repeatStride, preg_all_b16);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Reg::Add<half, Reg::MaskMergeMode::ZEROING>(vreg_exp_res, vreg_exp_res_1, vreg_exp_res_2, preg_all_b16);
            Reg::Cast<float, half, castTrait0>(vreg_exp_even, vreg_exp_res, preg_all_b16);
            Reg::Cast<float, half, castTrait1>(vreg_exp_odd, vreg_exp_res, preg_all_b16);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all_b32);
            Reg::ReduceSum<float, float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_sum, preg_all_b32);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(expSumUb, vreg_exp_sum, ureg_exp_sum, 1);
        }
        Reg::DataCopyUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(expSumUb, ureg_exp_sum, 0);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510NoUpdate8(
    const LocalTensor<T2> &dstTensor, const LocalTensor<float> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<float> &expMaxTensor, const LocalTensor<T> &inSrcTensor, const LocalTensor<float> &inExpSumTensor,
    const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &inMaskTensor, const LocalTensor<T> &inPseTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    // mode 1: originN = 128
    if constexpr (mode == 1) {
        SoftmaxFlashV510NoUpdateImpl128<T, T2, mode, sOuter, sInner>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    } else {
        SoftmaxFlashV510NoUpdateImpl256<T, T2, mode, sOuter, sInner>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    }
}

// originN = 128, Update
template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510UpdateImpl128(
    const LocalTensor<T2> &dstTensor, const LocalTensor<float> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<float> &expMaxTensor, const LocalTensor<T> &inSrcTensor, const LocalTensor<float> &inExpSumTensor,
    const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &inMaskTensor, const LocalTensor<T> &inPseTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 *expUb = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ float *expSumUb = (__ubuf__ float *)expSumTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)inSrcTensor.GetPhyAddr();
    __ubuf__ float *expMaxUb = (__ubuf__ float *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *inExpSumUb = (__ubuf__ float *)inExpSumTensor.GetPhyAddr();
    __ubuf__ T *inMaxUb = (__ubuf__ T *)inMaxTensor.GetPhyAddr();
    __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float *tmpExpSumUbStart = (__ubuf__ float *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T *tmpMaxUb = (__ubuf__ T *)((__ubuf__ float *)sharedTmpBuffer.GetPhyAddr() + 64);
    __ubuf__ T *tmpMaxUbStart = (__ubuf__ T *)((__ubuf__ float *)sharedTmpBuffer.GetPhyAddr() + 64);

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vreg_input_x;
        Reg::RegTensor<T> vreg_input_max;
        Reg::RegTensor<float> vreg_exp_sum;
        Reg::RegTensor<float> vreg_exp_sum_brc_even;
        Reg::RegTensor<float> vreg_exp_sum_brc_odd;
        Reg::RegTensor<T> vreg_in_max;
        Reg::RegTensor<T> vreg_max;
        Reg::RegTensor<T> vreg_exp_max;
        Reg::RegTensor<float> vreg_exp_max_even;
        Reg::RegTensor<float> vreg_exp_max_odd;
        Reg::RegTensor<float> vreg_in_exp_sum_even;
        Reg::RegTensor<float> vreg_in_exp_sum_odd;
        Reg::RegTensor<float> vreg_exp_sum_update_even;
        Reg::RegTensor<float> vreg_exp_sum_update_odd;
        Reg::RegTensor<T> vreg_exp_res;
        Reg::RegTensor<float> vreg_exp_even;
        Reg::RegTensor<float> vreg_exp_odd;

        Reg::UnalignReg ureg_max;
        Reg::UnalignReg ureg_exp_sum;

        Reg::MaskReg preg_all_b32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_all_b16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_all_b8 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_s8 = Reg::CreateMask<int8_t, Reg::MaskPattern::VL128>();

        Reg::RegTensor<half> vreg_cast_b16;
        Reg::RegTensor<half> vreg_cast_b16_unroll;
        Reg::RegTensor<half> vreg_cast_res;
        Reg::RegTensor<half> vreg_muls_res;
        // Reg::RegTensor<half> vregAddsRes;
        Reg::RegTensor<int8_t> vreg_cast;
        Reg::RegTensor<int8_t> vreg_res;

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x, srcUb + i * sInner);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_input_x, vreg_input_x, scale, preg_all_b16);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>(srcUb + i * sInner, vreg_input_x, preg_all_b16);
            Reg::ReduceMax<T, Reg::MaskMergeMode::ZEROING>(vreg_input_max, vreg_input_x, preg_all_b16);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, vreg_input_max, ureg_max, 1);
        }
        Reg::DataCopyUnAlignPost<T, Reg::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, ureg_max, 0);
        // load history max
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_in_max, inMaxUb);
        mem_bar(VST_VLD);
        // load current max
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_max, tmpMaxUbStart);
        // max(history max, current max)
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(vreg_max, vreg_input_max, vreg_in_max, preg_all_b16);
        // exp_max = exp(inmax - x_max)
        Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_exp_max, vreg_in_max, vreg_max,
                                                                                 preg_all_b16);
        Reg::Cast<float, half, castTrait0>(vreg_exp_max_even, vreg_exp_max, preg_all_b16);
        Reg::Cast<float, half, castTrait1>(vreg_exp_max_odd, vreg_exp_max, preg_all_b16);
        // store exp_max
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(expMaxUb, vreg_exp_max_even, vreg_exp_max_odd,
                                                             preg_all_b32);
        // store max
        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(maxUb, vreg_max, preg_all_b16);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B16>(vreg_max, maxUb + i);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x, srcUb + i * sInner);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_exp_res, vreg_input_x,
                                                                                     vreg_max, preg_all_b16);

            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vreg_muls_res, vreg_exp_res, (half)quantScaleP,
                                                               preg_all_b16);
            // Reg::Adds<half, half, Reg::MaskMergeMode::ZEROING>(vregAddsRes, vreg_muls_res, (half)offset,
            // preg_all_b16);

            Reg::Cast<int8_t, half, castTrait0>(vreg_cast, vreg_muls_res, preg_all_b16);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t> &)vreg_res,
                                                                   (Reg::RegTensor<uint16_t> &)vreg_cast);

            Reg::DataCopy<int8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ int8_t *&)expUb), vreg_res, blockStride, repeatStride, preg_s8);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Reg::Cast<float, half, castTrait0>(vreg_exp_even, vreg_exp_res, preg_all_b16);
            Reg::Cast<float, half, castTrait1>(vreg_exp_odd, vreg_exp_res, preg_all_b16);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all_b32);
            Reg::ReduceSum<float, float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_sum, preg_all_b32);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, vreg_exp_sum, ureg_exp_sum, 1);
        }
        Reg::DataCopyUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, ureg_exp_sum, 0);
        mem_bar(VST_VLD);

        // x_sum = sum(exp_max * in_sum + x_sum)
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vreg_in_exp_sum_even, vreg_in_exp_sum_odd, inExpSumUb);
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vreg_exp_sum_brc_even, vreg_exp_sum_brc_odd,
                                                             tmpExpSumUbStart);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum_update_even, vreg_exp_max_even, vreg_in_exp_sum_even,
                                                     preg_all_b32);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum_update_odd, vreg_exp_max_odd, vreg_in_exp_sum_odd,
                                                     preg_all_b32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum_update_even, vreg_exp_sum_update_even,
                                                     vreg_exp_sum_brc_even, preg_all_b32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum_update_odd, vreg_exp_sum_update_odd,
                                                     vreg_exp_sum_brc_odd, preg_all_b32);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(expSumUb, vreg_exp_sum_update_even,
                                                             vreg_exp_sum_update_odd, preg_all_b32);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510UpdateImpl256(
    const LocalTensor<T2> &dstTensor, const LocalTensor<float> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<float> &expMaxTensor, const LocalTensor<T> &inSrcTensor, const LocalTensor<float> &inExpSumTensor,
    const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &inMaskTensor, const LocalTensor<T> &inPseTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 *expUb1 = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T2 *expUb2 = (__ubuf__ T2 *)dstTensor.GetPhyAddr() + blockStride * 128;
    __ubuf__ float *expSumUb = (__ubuf__ float *)expSumTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)inSrcTensor.GetPhyAddr();
    __ubuf__ float *expMaxUb = (__ubuf__ float *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *inExpSumUb = (__ubuf__ float *)inExpSumTensor.GetPhyAddr();
    __ubuf__ T *inMaxUb = (__ubuf__ T *)inMaxTensor.GetPhyAddr();
    __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float *tmpExpSumUbStart = (__ubuf__ float *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T *tmpMaxUb = (__ubuf__ T *)((__ubuf__ float *)sharedTmpBuffer.GetPhyAddr() + 64);
    __ubuf__ T *tmpMaxUbStart = (__ubuf__ T *)((__ubuf__ float *)sharedTmpBuffer.GetPhyAddr() + 64);

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vreg_input_x_1;
        Reg::RegTensor<T> vreg_input_x_2;
        Reg::RegTensor<T> vreg_input_max;
        Reg::RegTensor<T> vreg_input_max_tmp;
        Reg::RegTensor<float> vreg_exp_sum;
        Reg::RegTensor<float> vreg_exp_sum_brc_even;
        Reg::RegTensor<float> vreg_exp_sum_brc_odd;
        Reg::RegTensor<T> vreg_in_max;
        Reg::RegTensor<T> vreg_max;
        Reg::RegTensor<T> vreg_exp_max;
        Reg::RegTensor<float> vreg_exp_max_even;
        Reg::RegTensor<float> vreg_exp_max_odd;
        Reg::RegTensor<float> vreg_in_exp_sum_even;
        Reg::RegTensor<float> vreg_in_exp_sum_odd;
        Reg::RegTensor<float> vreg_exp_sum_update_even;
        Reg::RegTensor<float> vreg_exp_sum_update_odd;
        Reg::RegTensor<T> vreg_exp_res;
        Reg::RegTensor<T> vreg_exp_res_1;
        Reg::RegTensor<T> vreg_exp_res_2;
        Reg::RegTensor<float> vreg_exp_even;
        Reg::RegTensor<float> vreg_exp_odd;

        Reg::UnalignReg ureg_max;
        Reg::UnalignReg ureg_exp_sum;

        Reg::MaskReg preg_all_b32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_all_b16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_all_b8 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg preg_s8 = Reg::CreateMask<int8_t, Reg::MaskPattern::VL128>();

        Reg::RegTensor<T> vreg_cast_b16;
        Reg::RegTensor<T> vreg_cast_b16_unroll;
        Reg::RegTensor<T> vreg_cast_res;
        Reg::RegTensor<T> vreg_muls_res;
        Reg::RegTensor<int8_t> vreg_cast;
        Reg::RegTensor<int8_t> vreg_res;

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x_1, srcUb + i * sInner);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb + i * sInner + halfRepSize);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_input_x_1, vreg_input_x_1, scale, preg_all_b16);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vreg_input_x_2, vreg_input_x_2, scale, preg_all_b16);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner, vreg_input_x_1, preg_all_b16);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner + halfRepSize, vreg_input_x_2,
                                                            preg_all_b16);
            Reg::Max(vreg_input_max_tmp, vreg_input_x_1, vreg_input_x_2, preg_all_b16);
            Reg::ReduceMax<T, Reg::MaskMergeMode::ZEROING>(vreg_input_max, vreg_input_max_tmp, preg_all_b16);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, vreg_input_max, ureg_max, 1);
        }
        Reg::DataCopyUnAlignPost<T, Reg::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, ureg_max, 0);
        // load history max
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_in_max, inMaxUb);
        mem_bar(VST_VLD);
        // load current max
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_max, tmpMaxUbStart);
        // max(history max, current max)
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(vreg_max, vreg_input_max, vreg_in_max, preg_all_b16);
        // exp_max = exp(inmax - x_max)
        Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_exp_max, vreg_in_max, vreg_max,
                                                                                 preg_all_b16);
        Reg::Cast<float, half, castTrait0>(vreg_exp_max_even, vreg_exp_max, preg_all_b16);
        Reg::Cast<float, half, castTrait1>(vreg_exp_max_odd, vreg_exp_max, preg_all_b16);
        // store exp_max
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(expMaxUb, vreg_exp_max_even, vreg_exp_max_odd,
                                                             preg_all_b32);
        // store max
        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(maxUb, vreg_max, preg_all_b16);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B16>(vreg_max, maxUb + i);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x_1, srcUb + i * sInner);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb + i * sInner + halfRepSize);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_exp_res_1, vreg_input_x_1,
                                                                                     vreg_max, preg_all_b16);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vreg_exp_res_2, vreg_input_x_2,
                                                                                     vreg_max, preg_all_b16);

            Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2 *&)expUb1, vreg_exp_res_1, blockStride, repeatStride, preg_all_b16);
            Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2 *&)expUb2, vreg_exp_res_2, blockStride, repeatStride, preg_all_b16);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Reg::Add<half, Reg::MaskMergeMode::ZEROING>(vreg_exp_res, vreg_exp_res_1, vreg_exp_res_2, preg_all_b16);
            Reg::Cast<float, half, castTrait0>(vreg_exp_even, vreg_exp_res, preg_all_b16);
            Reg::Cast<float, half, castTrait1>(vreg_exp_odd, vreg_exp_res, preg_all_b16);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all_b32);
            Reg::ReduceSum<float, float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_sum, preg_all_b32);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, vreg_exp_sum, ureg_exp_sum, 1);
        }
        Reg::DataCopyUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, ureg_exp_sum, 0);
        mem_bar(VST_VLD);

        // x_sum = sum(exp_max * in_sum + x_sum)
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vreg_in_exp_sum_even, vreg_in_exp_sum_odd, inExpSumUb);
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vreg_exp_sum_brc_even, vreg_exp_sum_brc_odd,
                                                             tmpExpSumUbStart);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum_update_even, vreg_exp_max_even, vreg_in_exp_sum_even,
                                                     preg_all_b32);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum_update_odd, vreg_exp_max_odd, vreg_in_exp_sum_odd,
                                                     preg_all_b32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum_update_even, vreg_exp_sum_update_even,
                                                     vreg_exp_sum_brc_even, preg_all_b32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum_update_odd, vreg_exp_sum_update_odd,
                                                     vreg_exp_sum_brc_odd, preg_all_b32);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(expSumUb, vreg_exp_sum_update_even,
                                                             vreg_exp_sum_update_odd, preg_all_b32);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510Update8(
    const LocalTensor<T2> &dstTensor, const LocalTensor<float> &expSumTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<float> &expMaxTensor, const LocalTensor<T> &inSrcTensor, const LocalTensor<float> &inExpSumTensor,
    const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &inMaskTensor, const LocalTensor<T> &inPseTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    // mode 1: originN = 128
    if constexpr (mode == 1) {
        SoftmaxFlashV510UpdateImpl128<T, T2, mode, sOuter, sInner>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    } else {
        SoftmaxFlashV510UpdateImpl256<T, T2, mode, sOuter, sInner>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    }
}

/*
 * @ingroup SoftmaxFlashV510
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] inExpSumTensor, in sum(exp(x-max)) of last softmax result
 * @param [in] inMaxTensor, in max value of last softmax result
 * @param [in] maskTensor, atten mask LocalTensor, each line padding to 32, padding value is 1
 * @param [in] pseTensor, reserved
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] originN, input origin colums, support range for sInner is: 0 < sInner <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isBmm2Concat, reserved
 * @param [in] isUpdate, enable flash mode
 * @param [in] mode
 *  mode 0: 64 < originN <= 128, and originN is not 8 aligned
 *  mode 1: 64 < originN <= 128, and originN is 8 aligned
 *  mode 2: 0 < originN <= 64
 * @param [in] hasAtten, indicates whether there is atten_mask
 * @param [in] hasPse, indicates whether there is pse_shift
 */
template <typename T, typename T2, bool isUpdate = false, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510_VF(const LocalTensor<T2> &dstTensor, const LocalTensor<float> &expSumTensor,
                                           const LocalTensor<T> &maxTensor, const LocalTensor<float> &expMaxTensor,
                                           const LocalTensor<T> &inSrcTensor, const LocalTensor<float> &inExpSumTensor,
                                           const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &inMaskTensor,
                                           const LocalTensor<T> &inPseTensor,
                                           const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t m,
                                           const uint32_t originN, const T scale, const T minValue, float quantScaleP)
{
    constexpr uint32_t blockU8 = 32;
    uint32_t blockN = 0;
    if constexpr (IsSameType<T2, int8_t>::value) {
        blockN = 32;
    } else {
        blockN = 16;
    }
    uint16_t blockStride = sOuter >> 1 | 0x1;

    if constexpr (!isUpdate) {
        SoftmaxFlashV510NoUpdate8<T, T2, mode, sOuter, sInner>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    } else {
        SoftmaxFlashV510Update8<T, T2, mode, sOuter, sInner>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    }
}
} // namespace FaVectorApi

#endif // MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_REGBASE_V2_INTERFACE_H
