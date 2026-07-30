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
 * \file vf_softmax.h
 * \brief
 */
#ifndef VF_SOFTMAX_H
#define VF_SOFTMAX_H

#include "kernel_tensor.h"
#include "vf_softmax_impl/vf_softmax_aligned128_no_update.h"
#include "vf_softmax_impl/vf_softmax_aligned128_update.h"


#ifdef __NPU_DEVICE__
namespace FaVectorApi {
#ifndef __CCE_KT_TEST__
/* **************************************************************************************************
 * Muls + Select(optional) + SoftmaxFlashV2 + Cast(fp32->fp16/bf16) + ND2NZ
 * ************************************************************************************************* */
using AscendC::LocalTensor;

enum OriginNRange {
    GT_64_AND_LTE_128,      // 64 < originN <= 128, support for non-alignment (s2BaseSize=128)
    EQ_128,                 // originN == 128, better performance than GT_64_AND_LTE_128 (s2BaseSize=128)
    GT_0_AND_LTE_64,        // 0 < originN <= 64 (s2BaseSize <= 64 or tail s2)
    N_INVALID
};

template <typename T, typename T2, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAttn = 0>
__aicore__ inline void ProcessVec1NoUpdate(
    const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const T scale, const T minValue)
{
    if constexpr (oriNRange == EQ_128) {
        LocalTensor<uint8_t> indexesTensor;
        ProcessVec1NoUpdateImpl128<T, T2, s1BaseSize, s2BaseSize, hasAttn>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, sharedTmpBuffer, m, originN, scale, minValue);
    } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
        LocalTensor<uint8_t> indexesTensor;
        ProcessVec1NoUpdateImpl64<T, T2, s1BaseSize, s2BaseSize, hasAttn>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, sharedTmpBuffer, m, originN, scale, minValue);
    } else { // GT_64_AND_LTE_128
        LocalTensor<uint8_t> indexesTensor;
        ProcessVec1NoUpdateGeneralImpl128<T, T2, s1BaseSize, s2BaseSize, hasAttn>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, sharedTmpBuffer, m, originN, scale, minValue);
    }
}

template <typename T, bool useNz = false>
__simd_vf__ inline void UpdateExpSumAndExpMaxImplVF(__ubuf__ T * maxUb, __ubuf__ T * inMaxUb, __ubuf__ T * expMaxUb,
    __ubuf__ T * expSumUb, __ubuf__ T * inExpSumUb, __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb, const uint32_t m)
{
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_in_exp_sum;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_exp_sum_brc;
    RegTensor<float> vreg_exp_sum_update;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    LoadAlign(vreg_max, tmpMaxUb);
    LoadAlign(vreg_in_max, inMaxUb);
    ExpSub(vreg_exp_max, vreg_in_max, vreg_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)maxUb, vreg_max, preg_all);

    LoadAlign(vreg_in_exp_sum, inExpSumUb);
    LoadAlign(vreg_exp_sum_brc, tmpExpSumUb);
    Mul(vreg_exp_sum_update, vreg_exp_max, vreg_in_exp_sum, preg_all);
    Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_exp_sum_brc, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)expSumUb, vreg_exp_sum_update, preg_all);
}

template <typename T, bool useNz = false>
__aicore__ inline void UpdateExpSumAndExpMaxImpl(
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor,  const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m)
{
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();

    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();

    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    uint32_t repSize = floatRepSize;
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + repSize;
    UpdateExpSumAndExpMaxImplVF<T, useNz>(maxUb, inMaxUb, expMaxUb, expSumUb, inExpSumUb, tmpExpSumUb, tmpMaxUb, m);
}

template <typename T, bool useNz = false>
__aicore__ inline void UpdateExpSumAndExpMax(const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m)
{
    UpdateExpSumAndExpMaxImpl<T, useNz>(expSumTensor, maxTensor, expMaxTensor,
                                 inExpSumTensor, inMaxTensor, sharedTmpBuffer, m);
}

template <typename T, typename T2, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAttn = 0>
__aicore__ inline void ProcessVec1Update(
    const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const T scale, const T minValue)
{
    if constexpr (oriNRange == EQ_128) {
        LocalTensor<uint8_t> indexesTensor;
        ProcessVec1UpdateImpl128<T, T2, s1BaseSize, s2BaseSize, hasAttn>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, sharedTmpBuffer, m, originN, scale, minValue);
    } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
        LocalTensor<uint8_t> indexesTensor;
        ProcessVec1UpdateImpl64<T, T2, s1BaseSize, s2BaseSize, hasAttn>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, sharedTmpBuffer, m, originN, scale, minValue);
    } else { // GT_64_AND_LTE_128
        LocalTensor<uint8_t> indexesTensor;
        ProcessVec1UpdateGeneralImpl128<T, T2, s1BaseSize, s2BaseSize, hasAttn>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, sharedTmpBuffer, m, originN, scale, minValue);
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
 * @param [in] inExpSumTensor, in sum(exp(x-max)) of last softmax result
 * @param [in] inMaxTensor, in max value of last softmax result
 * @param [in] maskTensor, attn mask LocalTensor, each line padding to 32, padding value is 1
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] s2BaseSize, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 * @param [in] hasAttn, indicates whether there is attn_mask
 */
template <typename T, typename T2, bool isUpdate = false, uint32_t s1BaseSize = 128,
          uint32_t s2BaseSize = 128, OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAttn = 0>
__aicore__ inline void ProcessVec1Vf(const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf,
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint16_t m, const uint32_t originN, const T scale, const T minValue)
{
    static_assert(IsSameType<T, float>::value, "VF softmax, T must be float");
    static_assert(IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value,
                  "VF softmax, T2 must be half or bfloat16");
    static_assert(oriNRange < N_INVALID, "VF softmax, oriNRange is invalid");

    if constexpr (!isUpdate) {
        ProcessVec1NoUpdate<T, T2, s1BaseSize, s2BaseSize, oriNRange, hasAttn>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, sharedTmpBuffer,
            m, originN, scale, minValue);
    } else {
        ProcessVec1Update<T, T2, s1BaseSize, s2BaseSize, oriNRange, hasAttn>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, sharedTmpBuffer,
            m, originN, scale, minValue);
    }
}


#endif
} // namespace
#endif
#endif // VF_SOFTMAX_H
