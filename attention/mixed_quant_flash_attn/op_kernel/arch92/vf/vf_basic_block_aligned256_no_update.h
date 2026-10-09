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
 * \file vf_basic_block_aligned256_no_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_ALIGNED256_NO_UPDATE_H
#define VF_BASIC_BLOCK_ALIGNED256_NO_UPDATE_H

#include "mqfa_vf_basic_block_utils.h"
#include "mqfa_pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool hasSink = false>
__simd_vf__ void ProcessVec1NoUpdateAlignImpl256VF(
    __ubuf__ T2* expUb1, __ubuf__ T2* expUb2, __ubuf__ pseShiftType* pseUb, __ubuf__ T* expSumUb, __ubuf__ T* maxUb,
    __ubuf__ T* maxUbStart, __ubuf__ T* srcUb, __ubuf__ uint32_t* maskUb1, __ubuf__ uint32_t* maskUb2,
    __ubuf__ uint32_t* maskUb3, __ubuf__ uint32_t* maskUb4, __ubuf__ uint32_t* dropMaskUb1,
    __ubuf__ uint32_t* dropMaskUb2, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride,
    const uint32_t oriTailN1, const uint32_t oriTailN2, const uint32_t tailN1, const uint32_t tailN2,
    uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2, float divValue,
    const uint16_t m, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const T minValue, const float sinkValue)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_sel1;
    RegTensor<float> vreg_sel2;
    RegTensor<float> vreg_sel3;
    RegTensor<float> vreg_sel4;
    RegTensor<float> vreg_input_x1;
    RegTensor<float> vreg_input_x2;
    RegTensor<float> vreg_input_x3;
    RegTensor<float> vreg_input_x4;
    RegTensor<float> vreg_max_tmp1;
    RegTensor<float> vreg_max_tmp2;
    RegTensor<float> vreg_max_tmp3;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max_brc;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_exp_sum1;
    RegTensor<float> vreg_exp_sum2;
    RegTensor<float> vreg_exp_sum3;
    RegTensor<float> vreg_exp_even1;
    RegTensor<float> vreg_exp_odd1;
    RegTensor<float> vreg_exp_even2;
    RegTensor<float> vreg_exp_odd2;
    RegTensor<float> vreg_sel_drop;
    RegTensor<float> vreg_sel_drop2;
    RegTensor<float> vreg_sink_input;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_exp_even1_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
    RegTensor<bfloat16_t> vreg_exp_even2_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
    RegTensor<bfloat16_t> vreg_exp1_bf16;
    RegTensor<bfloat16_t> vreg_exp2_bf16;
    // half
    RegTensor<half> vreg_exp_even1_f16;
    RegTensor<half> vreg_exp_odd1_f16;
    RegTensor<half> vreg_exp_even2_f16;
    RegTensor<half> vreg_exp_odd2_f16;
    RegTensor<half> vreg_exp1_f16;
    RegTensor<half> vreg_exp2_f16;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();

    MaskReg preg_compare1;
    MaskReg preg_compare2;
    MaskReg preg_compare3;
    MaskReg preg_compare4;

    MaskReg preg1;
    MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();

    MaskReg preg3;
    MaskReg preg4;
    MaskReg preg5;
    MaskReg preg6;

    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x1, srcUb + i * s2BaseSize);
        LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
        LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
        LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);

        if constexpr (hasAtten == 1) {
            // atten mask
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare1, (__ubuf__ uint32_t*&)maskUb1, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare2, (__ubuf__ uint32_t*&)maskUb2, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare3, (__ubuf__ uint32_t*&)maskUb3, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare4, (__ubuf__ uint32_t*&)maskUb4, nPadding);
            Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
            Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
            Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
            Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)srcUb + i * s2BaseSize, vreg_sel1,
                                                              preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_sel2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)srcUb + floatRepSize * 2 + i * s2BaseSize,
                                                              vreg_sel3, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)srcUb + floatRepSize * 3 + i * s2BaseSize,
                                                              vreg_sel4, preg_all);
            Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
            Max(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all);
            Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                              vreg_max_tmp3, preg_all);
        } else {
            Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
            Max(vreg_max_tmp2, vreg_input_x3, vreg_input_x4, preg_all);
            Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                              vreg_max_tmp3, preg_all);
        }
        if constexpr (hasSink) {
            Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
        }
        StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T*&)maxUb), vreg_input_max, ureg_max, 1);
    }
    StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T*&)maxUb), ureg_max, 0);
    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, maxUbStart + i);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x3, vreg_input_x4,
                                                          srcUb + floatRepSize * 2 + i * s2BaseSize);
        ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
        Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
        Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum3, vreg_exp_sum3,
                                                                                          preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T*&)expSumUb), vreg_exp_sum3,
                                                                     ureg_exp_sum, 1);
        // dropmask compute
        if constexpr (hasDrop == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg1, (__ubuf__ uint32_t*&)dropMaskUb1, s2BaseSize >> 3);
            // preg1: 0011223344556677 preg2: 0000000000000000
            MaskInterleave<half>(preg3, preg4, preg1, preg2);
            // preg3: 0000110022003300 preg4: 4400550066007700
            MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg1, (__ubuf__ uint32_t*&)dropMaskUb2, s2BaseSize >> 3);
            // preg1: 0011223344556677 preg2: 0000000000000000
            MaskInterleave<half>(preg3, preg4, preg1, preg2);
            // preg3: 0000110022003300 preg4: 4400550066007700
            MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
        }

        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
               (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
               (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
        } else if constexpr (IsSameType<T2, half>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
            Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16,
               (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16,
               (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_all_b16);
            // fp8_e5m2_t
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T*&)expSumUb), ureg_exp_sum, 0);
}

// no update, 128 < originN <= 256
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool hasSink = false>
__aicore__ inline void ProcessVec1NoUpdateAlignImpl256(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const T minValue, float keepProb, const float sinkValue)
{
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (65*16*2/32)，单位block, low 16bits: repeatStride (1)
    const uint32_t blockStride = (s1BaseSize / ArchInfo::CV_RATIO) | 0x1;
    const uint32_t repeatStride = 1;
    __ubuf__ T2* expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2* expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize / ArchInfo::CV_RATIO) + 1) * (128);
    __ubuf__ pseShiftType* pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ uint32_t* maskUb1 = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint32_t* maskUb2 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize);
    __ubuf__ uint32_t* maskUb3 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize * 2);
    __ubuf__ uint32_t* maskUb4 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize * 3);
    __ubuf__ uint32_t* dropMaskUb1 = (__ubuf__ uint32_t*)dropTensor.GetPhyAddr();
    __ubuf__ uint32_t* dropMaskUb2 = (__ubuf__ uint32_t*)(dropTensor.GetPhyAddr() + s2BaseSize / 16);

    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t oriTailN1 = originN - floatRepSize * 2 < floatRepSize ? originN - floatRepSize * 2 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 3) <= 0 ? 0 : originN - floatRepSize * 3;
    const uint32_t tailN1 = s2BaseSize - floatRepSize * 2;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 3;
    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    float divValue = 1.0f / keepProb;

    ProcessVec1NoUpdateAlignImpl256VF<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, hasSink>(
        expUb1, expUb2, pseUb, expSumUb, maxUb, maxUbStart, srcUb, maskUb1, maskUb2, maskUb3, maskUb4, dropMaskUb1,
        dropMaskUb2, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2, tailN1, tailN2, pltOriTailN1,
        pltOriTailN2, pltTailN1, pltTailN2, divValue, m, pseStride, slopes, posShift, scale, minValue, sinkValue);
}
} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_UNALIGNED256_NO_UPDATE_H
