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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_mxfp8.h
 * \brief
 */
#ifndef MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_MXFP8_H
#define MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_MXFP8_H

#include "kernel_tensor.h"

#include "vf_basic_block_unaligned256_no_update_mxfp8.h"
#include "vf_basic_block_unaligned256_update_mxfp8.h"

using namespace regbaseutil;

namespace FaVectorApi {
/* **************************************************************************************************
 * Muls + Select(optional) + SoftmaxFlashV2 + Cast(fp32->fp16/bf16) + ND2NZ
 * ************************************************************************************************* */
using AscendC::LocalTensor;

enum OriginNRange {
    GT_128_AND_LTE_256 = 0, // 128 < originN <= 256, support for non-alignment (s2BaseSize=256)
    GT_64_AND_LTE_128,      // 64 < originN <= 128, support for non-alignment (s2BaseSize=128)
    EQ_128,                 // originN == 128, better performance than GT_64_AND_LTE_128 (s2BaseSize=128)
    GT_0_AND_LTE_64,        // 0 < originN <= 64 (s2BaseSize <= 64 or tail s2)
    GT_256_AND_LTE_512,     // 256 < originN <= 512 (s2BaseSize <= 512 or tail s2)
    GT_512_AND_LTE_1024,    // 512 < originN <= 1024 (s2BaseSize <= 1024 or tail s2)
    GT_0_AND_LTE_256,       // 0 < originN <= 256 (s2BaseSize <= 256 or tail s2)
    N_INVALID
};
// ==================== Mxfp8 ====================
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
          OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0,
          PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1NoUpdateMxfp8(
    const LocalTensor<T2> &dstTensor, const LocalTensor<uint8_t> &vselrIndexesBuf, const LocalTensor<T> &expSumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inExpSumTensor, const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &maskTensor,
    const LocalTensor<pseShiftType> &pseTensor, const LocalTensor<uint8_t> &dropTensor,
    const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const LocalTensor<float> &preLoopMaxTensor, const LocalTensor<float> &preLoopSumTensor,
    const LocalTensor<float> &firstLoopSumTensor, uint32_t subLoop, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb, const LocalTensor<T> &queryScaleUb = LocalTensor<T>(),
    const float deSCaleKValue = 1.0f, const float pScale = 1.0f)
{
    LocalTensor<uint8_t> indexesTensor;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)];
    }
    ProcessVec1NoUpdateGeneralImpl256Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                                                    hasDrop, false>(
        dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
        maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor, sharedTmpBuffer, subLoop, m, originN, pseStride,
        slopes, posShift, scale, dScaleQK, minValue, keepProb, 0.0F, pScale);
}

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
          OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0,
          PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1UpdateMxfp8(
    const LocalTensor<T2> &dstTensor, const LocalTensor<uint8_t> &vselrIndexesBuf, const LocalTensor<T> &expSumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inExpSumTensor, const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &maskTensor,
    const LocalTensor<pseShiftType> &pseTensor, const LocalTensor<uint8_t> &dropTensor,
    const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const LocalTensor<float> &preLoopMaxTensor, const LocalTensor<float> &preLoopSumTensor,
    const LocalTensor<float> &firstLoopSumTensor, uint32_t subLoop, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb, const LocalTensor<T> &queryScaleUb = LocalTensor<T>(),
    const float deSCaleKValue = 1.0f, const float pScale = 1.0f)
{
    LocalTensor<uint8_t> indexesTensor;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)];
    }
    ProcessVec1UpdateGeneralImpl256Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                                                  hasDrop, false>(
        dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
        maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor, sharedTmpBuffer, preLoopMaxTensor, preLoopSumTensor,
        firstLoopSumTensor, subLoop, m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, 0.0F,
        pScale);
}

template <typename T, typename T2, typename pseShiftType, bool isUpdate = false, uint32_t s1BaseSize = 128,
          uint32_t s2BaseSize = 128, OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0,
          PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1VfMxfp8(
    const LocalTensor<T2> &dstTensor, const LocalTensor<uint8_t> &vselrIndexesBuf, const LocalTensor<T> &expSumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inExpSumTensor, const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &maskTensor,
    const LocalTensor<pseShiftType> &pseTensor, const LocalTensor<uint8_t> &dropTensor,
    const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const LocalTensor<T> &pScaleTensor, const LocalTensor<float> &preLoopMaxTensor,
    const LocalTensor<float> &preLoopSumTensor, const LocalTensor<float> &firstLoopSumTensor, uint32_t subLoop,
    const uint16_t m, const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift,
    const T scale, const float dScaleQK, const T minValue, float keepProb,
    const LocalTensor<T> &queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f, const float pScale = 1.0f)
{
    static_assert(IsSameType<T, float>::value, "VF mul_sel_softmaxflashv2_cast_nz, T must be float");
    static_assert(IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value || IsSameType<T2, float>::value ||
                      IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value,
                  "VF mul_sel_softmaxflashv2_cast_nz, T2 must be half, bfloat16 or float or fp8 or int8");
    static_assert(oriNRange < N_INVALID, "VF mul_sel_softmaxflashv2_cast_nz, oriNRange is invalid");

    if constexpr (!isUpdate) {
        ProcessVec1NoUpdateMxfp8<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode, hasDrop,
                                 isMlaSgd>(dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                                           inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor,
                                           pScaleSubLoop0Tensor, sharedTmpBuffer, preLoopMaxTensor, preLoopSumTensor,
                                           firstLoopSumTensor, subLoop, m, originN, pseStride, slopes, posShift, scale,
                                           dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, pScale);
    } else {
        ProcessVec1UpdateMxfp8<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode, hasDrop,
                               isMlaSgd>(dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                                         inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor,
                                         pScaleSubLoop0Tensor, sharedTmpBuffer, preLoopMaxTensor, preLoopSumTensor,
                                         firstLoopSumTensor, subLoop, m, originN, pseStride, slopes, posShift, scale,
                                         dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, pScale);
    }
}
} // namespace FaVectorApi

#endif // MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_MXFP8_H
