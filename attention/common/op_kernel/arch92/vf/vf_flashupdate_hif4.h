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
 * \file vf_flashupdate_hif4.h
 * \brief
 */
#ifndef VF_FLASHUPDATE_HIF4_H
#define VF_FLASHUPDATE_HIF4_H

#include "kernel_tensor.h"

namespace FaVectorApi {

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize>
__simd_vf__ inline void FlashUpdateBasicVF(__ubuf__ float* dstUb, __ubuf__ float* curUb, __ubuf__ float* preUb,
                                           __ubuf__ half* expMaxUb, __ubuf__ float* preExpSumUb,
                                           __ubuf__ float* expSumUb, const uint16_t m, const uint16_t d)
{
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize; // 几轮循环

    RegTensor<half> vreg_exp_max_half; // 临时FP16寄存器 (加载用)
    RegTensor<half> vreg_exp_max_half_0;
    RegTensor<half> vreg_exp_max_half_1;
    RegTensor<float> vreg_exp_max; // 转换后的FP32寄存器 (计算用)
    RegTensor<float> vreg_exp_max_0;
    RegTensor<float> vreg_exp_max_1;
    RegTensor<float> vreg_pre_exp_sum_0;
    RegTensor<float> vreg_pre_exp_sum_1;
    RegTensor<float> vreg_exp_sum_0;
    RegTensor<float> vreg_exp_sum_1;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<half, MaskPattern::ALL>();

    // dstTensor = preTensor * expMaxTensor + curTensor
    for (uint16_t i = 0; i < m; ++i) {
        DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(vreg_exp_max_half, expMaxUb + i * reduceSize); // [m,8]
        Cast<float, half, castTraitFp16_32_update>(vreg_exp_max, vreg_exp_max_half, preg_all_b16);      // FP16 → FP32
        for (uint16_t j = 0; j < dLoops; ++j) {
            DataCopy(vreg_input_pre, preUb + i * d + j * floatRepSize);
            DataCopy(vreg_input_cur, curUb + i * d + j * floatRepSize);

            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)dstUb + i * d + j * floatRepSize,
                                                                vreg_add, preg_all);
        }
    }
    // vreg_exp_max*sum sum更新
    DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(vreg_exp_max_half, expMaxUb);
    Interleave(vreg_exp_max_half_0, vreg_exp_max_half_1, vreg_exp_max_half, vreg_exp_max_half);
    Cast<float, half, castTraitFp16_32_update>(vreg_exp_max_0, vreg_exp_max_half_0, preg_all_b16); // FP16 → FP32
    Cast<float, half, castTraitFp16_32_update>(vreg_exp_max_1, vreg_exp_max_half_1, preg_all_b16);
    DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vreg_pre_exp_sum_0, preExpSumUb); // old
    DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vreg_pre_exp_sum_1, preExpSumUb + 64);
    DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vreg_exp_sum_0, expSumUb); // new
    DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vreg_exp_sum_1, expSumUb + 64);
    Mul(vreg_pre_exp_sum_0, vreg_pre_exp_sum_0, vreg_exp_max_0, preg_all); // old*exp +new
    Mul(vreg_pre_exp_sum_1, vreg_pre_exp_sum_1, vreg_exp_max_1, preg_all); // old*exp +new
    Add(vreg_exp_sum_0, vreg_pre_exp_sum_0, vreg_exp_sum_0, preg_all);
    Add(vreg_exp_sum_1, vreg_pre_exp_sum_1, vreg_exp_sum_1, preg_all);
    DataCopy<float, MicroAPI::StoreDist::DIST_NORM>((__ubuf__ float*&)preExpSumUb, vreg_exp_sum_0, preg_all);
    DataCopy<float, MicroAPI::StoreDist::DIST_NORM>((__ubuf__ float*&)preExpSumUb + 64, vreg_exp_sum_1, preg_all);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize>
__aicore__ inline void FlashUpdateBasic(const uint64_t& dstTensor, const LocalTensor<float>& curTensor,
                                        const uint64_t& preTensor, const uint64_t& expMaxTensor,
                                        const LocalTensor<float>& preExpSumTensor,
                                        const LocalTensor<float>& expSumTensor, const uint16_t m, const uint16_t d)
{
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor;
    __ubuf__ float* curUb = (__ubuf__ float*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ float*)preTensor;
    __ubuf__ half* expMaxUb = (__ubuf__ half*)expMaxTensor;
    // sum dn的时候要搬进来更新
    __ubuf__ float* preExpSumUb = (__ubuf__ float*)preExpSumTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();

    FlashUpdateBasicVF<T, INPUT_T, OUTPUT_T, srcD, reduceSize>(dstUb, curUb, preUb, expMaxUb, preExpSumUb, expSumUb, m,
                                                               d);
}

// dstTensor = preTensor * expMaxTensor + curTensor
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void FlashUpdateNew(const uint64_t& dstTensor, const LocalTensor<float>& curTensor,
                                      const uint64_t& preTensor, const uint64_t& expMaxTensor,
                                      const LocalTensor<float>& preExpSumTensor, const LocalTensor<float>& expSumTensor,
                                      const uint16_t m, const uint16_t d)
{
    FlashUpdateBasic<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE>(dstTensor, curTensor, preTensor, expMaxTensor,
                                                              preExpSumTensor, expSumTensor, m, d);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize>
__simd_vf__ inline void FlashUpdateLastBasicVF(__ubuf__ float* dstUb, __ubuf__ float* curUb, __ubuf__ float* preUb,
                                               __ubuf__ half* expMaxUb, __ubuf__ float* preExpSumUb,
                                               __ubuf__ float* expSumUb, const uint16_t m, const uint16_t d)
{
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    RegTensor<half> vreg_exp_max_half;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_pre_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<half, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(vreg_exp_max_half, expMaxUb + i * reduceSize);
        DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);
        DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_pre_exp_sum, preExpSumUb + i * reduceSize);
        Cast<float, half, castTraitFp16_32_update>(vreg_exp_max, vreg_exp_max_half, preg_all_b16); // FP16 → FP32
        Mul(vreg_pre_exp_sum, vreg_pre_exp_sum, vreg_exp_max, preg_all);
        Add(vreg_exp_sum, vreg_exp_sum, vreg_pre_exp_sum, preg_all);
        Store((__ubuf__ float*&)preExpSumUb + i, vreg_exp_sum, 1);
        for (uint16_t j = 0; j < dLoops; ++j) {
            DataCopy(vreg_input_pre, preUb + i * d + j * floatRepSize);
            DataCopy(vreg_input_cur, curUb + i * d + j * floatRepSize);

            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);

            DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)dstUb + i * d + j * floatRepSize,
                                                                vreg_div, preg_all);
        }
    }
}
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize>
__aicore__ inline void FlashUpdateLastBasic(const uint64_t& dstTensor, const LocalTensor<float>& curTensor,
                                            const uint64_t& preTensor, const uint64_t& expMaxTensor,
                                            const LocalTensor<float>& preExpSumTensor,
                                            const LocalTensor<float>& expSumTensor, const uint16_t m, const uint16_t d)
{
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor;
    __ubuf__ float* curUb = (__ubuf__ float*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ float*)preTensor;
    __ubuf__ half* expMaxUb = (__ubuf__ half*)expMaxTensor;
    // sum dn的时候要搬进来更新
    __ubuf__ float* preExpSumUb = (__ubuf__ float*)preExpSumTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();

    FlashUpdateLastBasicVF<T, INPUT_T, OUTPUT_T, srcD, reduceSize>(dstUb, curUb, preUb, expMaxUb, preExpSumUb, expSumUb,
                                                                   m, d);
}

// dstTensor = (preTensor * expMaxTensor + curTensor) / expSumTensor
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void FlashUpdateLastNew(const uint64_t& dstTensor, const LocalTensor<float>& curTensor,
                                          const uint64_t& preTensor, const uint64_t& expMaxTensor,
                                          const LocalTensor<float>& preExpSumTensor,
                                          const LocalTensor<float>& expSumTensor, uint16_t m, uint16_t d)
{
    FlashUpdateLastBasic<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE>(dstTensor, curTensor, preTensor, expMaxTensor,
                                                                  preExpSumTensor, expSumTensor, m, d);
}
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD>
__simd_vf__ inline void LastDivNewVF(__ubuf__ float* dstUb, __ubuf__ float* preUb, __ubuf__ float* expSumUb,
                                     const uint16_t m, const uint16_t d)
{
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_div;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * REDUCE_SIZE);
        for (uint16_t j = 0; j < dLoops; ++j) {
            DataCopy(vreg_input_pre, preUb + i * d + j * floatRepSize);
            Div(vreg_div, vreg_input_pre, vreg_exp_sum, preg_all);

            DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)dstUb + i * d + j * floatRepSize,
                                                                vreg_div, preg_all);
        }
    }
}
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void LastDivNew(const uint64_t& dstTensor, const uint64_t& preTensor,
                                  const LocalTensor<float>& expSumTensor, uint16_t m, uint16_t d)
{
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor;
    __ubuf__ float* preUb = (__ubuf__ float*)preTensor;
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    LastDivNewVF<T, INPUT_T, OUTPUT_T, srcD>(dstUb, preUb, expSumUb, m, d);
}

template <typename T>
__simd_vf__ inline void ComputeLseOutputVF(__ubuf__ half* srcMaxUb, __ubuf__ T* srcSumUb, __ubuf__ T* dstUb,
                                           const uint32_t dealCount, const uint32_t min = 0xFF7FFFFF)
{
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<half> vregMaxHalf;
    MicroAPI::RegTensor<T> vregMax;
    MicroAPI::RegTensor<T> vregRes;
    MicroAPI::RegTensor<T> vregResFinal;
    MicroAPI::RegTensor<float> vregMinValue;
    MicroAPI::RegTensor<float> vregInfValue;
    MicroAPI::MaskReg pregCompare;
    constexpr uint32_t dealRows = 8;
    constexpr uint32_t floatRepSize = 64; // 64: 一个寄存器存64个float
    constexpr float infValue = 3e+99;     // 3e+99 for float inf
    const float minValue = *((float*)&min);
    uint16_t updateLoops = dealCount / dealRows;
    uint16_t tailLSize = dealCount % dealRows * 8;
    uint32_t pltTail = static_cast<uint32_t>(tailLSize);
    uint32_t pltTail_half = static_cast<uint32_t>(tailLSize) * 2;

    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg preg_all_b16 = CreateMask<half, MaskPattern::ALL>();
    MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<T>(pltTail);
    MicroAPI::MaskReg preg_all_b16_tail = MicroAPI::UpdateMask<half>(pltTail_half);
    MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
    MicroAPI::Duplicate<float, float>(vregInfValue, infValue);

    for (uint16_t i = 0; i < updateLoops; ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregSum, srcSumUb + (i * dealRows));
        MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_E2B_B16>(vregMaxHalf, srcMaxUb + (i * dealRows));
        // cast
        Cast<T, half, castTraitFp16_32_update>(vregMax, vregMaxHalf, preg_all_b16);

        MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSum, pregAll);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregAll);

        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregAll);
        MicroAPI::Select<T>(vregResFinal, vregInfValue, vregRes, pregCompare);

        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + (i * floatRepSize), vregResFinal, pregAll);
    }

    if (tailLSize != 0) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregSum, srcSumUb + dealRows * updateLoops);
        MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_E2B_B16>(vregMaxHalf, srcMaxUb + dealRows * updateLoops);

        Cast<T, half, castTraitFp16_32_update>(vregMax, vregMaxHalf, preg_all_b16_tail);

        MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSum, pregTail);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregTail);

        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregTail);
        MicroAPI::Select<T>(vregResFinal, vregInfValue, vregRes, pregCompare);

        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + floatRepSize * updateLoops, vregResFinal,
                                                                    pregTail);
    }
}

__aicore__ inline void ComputeLseOutputVF(const LocalTensor<float>& dstTensor, const uint64_t& softmaxMaxTensor, // half
                                          LocalTensor<float>& softmaxSumTensor, uint32_t dealCount,
                                          const uint32_t min = 0xFF7FFFFF)
{
    __ubuf__ half* srcMaxUb = (__ubuf__ half*)softmaxMaxTensor;
    __ubuf__ float* srcSumUb = (__ubuf__ float*)softmaxSumTensor.GetPhyAddr();
    __ubuf__ float* dstUb = (__ubuf__ float*)dstTensor.GetPhyAddr();

    ComputeLseOutputVF<float>(srcMaxUb, srcSumUb, dstUb, dealCount, min);
}

} // namespace FaVectorApi

#endif // VF_FLASHUPDATE_HIF4_H
