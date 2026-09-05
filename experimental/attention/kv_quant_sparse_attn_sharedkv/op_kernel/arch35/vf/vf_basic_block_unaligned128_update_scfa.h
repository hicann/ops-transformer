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
 * \file vf_basic_block_unaligned128_update_scfa.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED128_UPDATE_SCFA_H
#define VF_BASIC_BLOCK_UNALIGNED128_UPDATE_SCFA_H

#include "vf_basic_block_utils.h"
#include "../util_regbase.h"
#include "../kv_quant_sparse_attn_sharedkv_common_arch35.h"

using namespace regbaseutil;

namespace SCFaVectorApi {

template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128>
__simd_vf__ void ProcessVec1UpdateGeneralImpl128VF(__ubuf__ T2 *expUb, __ubuf__ T *srcUb, __ubuf__ T *inMaxUb,
                                                   __ubuf__ T *tmpExpSumUb, __ubuf__ T *tmpMaxUb, __ubuf__ T *tmpMaxUb2,
                                                   const uint32_t blockStride, const uint32_t repeatStride,
                                                   const uint16_t m, const T scale, const T minValue,
                                                   uint32_t pltOriTailN, uint32_t pltTailN, uint32_t pltN)
{
    AscendC::Reg::RegTensor<float> vreg_min;
    AscendC::Reg::RegTensor<float> vreg_input_x;
    AscendC::Reg::RegTensor<float> vreg_input_x_unroll;
    AscendC::Reg::RegTensor<float> vreg_input_x_unroll_new;
    AscendC::Reg::RegTensor<float> vreg_max_tmp;
    AscendC::Reg::RegTensor<float> vreg_cur_max;
    AscendC::Reg::RegTensor<float> vreg_max_new;
    AscendC::Reg::RegTensor<float> vreg_exp_sum;
    AscendC::Reg::RegTensor<float> vreg_in_max;
    AscendC::Reg::RegTensor<float> vreg_max_brc;
    AscendC::Reg::RegTensor<float> vreg_exp_even;
    AscendC::Reg::RegTensor<float> vreg_exp_odd;

    // bfloat16_t
    AscendC::Reg::RegTensor<bfloat16_t> vreg_exp_even_bf16;
    AscendC::Reg::RegTensor<bfloat16_t> vreg_exp_odd_bf16;
    AscendC::Reg::RegTensor<bfloat16_t> vreg_exp_bf16;
    AscendC::Reg::RegTensor<bfloat16_t> vreg_pse_bf16_src;
    AscendC::Reg::RegTensor<bfloat16_t> vreg_pse_bf16;
    AscendC::Reg::RegTensor<bfloat16_t> vreg_pse_bf16_unroll;

    AscendC::Reg::UnalignRegForStore ureg_max;
    AscendC::Reg::UnalignRegForStore ureg_exp_sum;

    AscendC::Reg::MaskReg preg_all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg preg_all_b16 = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg preg_n_b16 = AscendC::Reg::UpdateMask<uint16_t>(pltN);
    AscendC::Reg::MaskReg preg_tail_n = AscendC::Reg::UpdateMask<T>(pltTailN);
    AscendC::Reg::MaskReg preg_ori_tail_n = AscendC::Reg::UpdateMask<T>(pltOriTailN);

    AscendC::Reg::Duplicate(vreg_min, minValue);
    // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
    for (uint16_t i = 0; i < m; ++i) {
        AscendC::Reg::LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        AscendC::Reg::LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
        AscendC::Reg::Muls(vreg_input_x, vreg_input_x, scale, preg_all); // Muls(scale)
        AscendC::Reg::Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_ori_tail_n);
        AscendC::Reg::Select(vreg_input_x_unroll_new, vreg_input_x_unroll, vreg_min, preg_ori_tail_n);
        AscendC::Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x,
                                                                   preg_all);
        AscendC::Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                                   vreg_input_x_unroll_new, preg_tail_n);
        AscendC::Reg::Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll_new, preg_all);
        AscendC::Reg::Reduce<Reg::ReduceType::MAX, float, float, Reg::MaskMergeMode::ZEROING>(vreg_cur_max,
                                                                                              vreg_max_tmp, preg_all);

        AscendC::Reg::StoreUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), vreg_cur_max,
                                                                              ureg_max, 1);
    }
    AscendC::Reg::StoreUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
    AscendC::Reg::LoadAlign(vreg_in_max, inMaxUb);
    AscendC::Reg::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign(vreg_cur_max, tmpMaxUb2);                     // 获取新的max[s1, 1]
    AscendC::Reg::Max(vreg_max_new, vreg_cur_max, vreg_in_max, preg_all); // 计算新、旧max的最大值
    AscendC::Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
    AscendC::Reg::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        AscendC::Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_max_brc, tmpMaxUb2 + i);
        AscendC::Reg::LoadAlign<T, Reg::LoadDist::DIST_DINTLV_B32>(vreg_input_x, vreg_input_x_unroll,
                                                                   srcUb + i * s2BaseSize);
        AscendC::Reg::ExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc, preg_all);
        AscendC::Reg::ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max_brc, preg_all);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        AscendC::Reg::Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
        AscendC::Reg::Reduce<Reg::ReduceType::SUM, float, float, Reg::MaskMergeMode::ZEROING>(vreg_exp_sum,
                                                                                              vreg_exp_sum, preg_all);
        AscendC::Reg::StoreUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb),
                                                                              vreg_exp_sum, ureg_exp_sum, 1);

        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            AscendC::Reg::Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
            AscendC::Reg::Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
            AscendC::Reg::Or((RegTensor<uint16_t> &)vreg_exp_bf16, (RegTensor<uint16_t> &)vreg_exp_even_bf16,
                             (RegTensor<uint16_t> &)vreg_exp_odd_bf16, preg_all_b16);
            AscendC::Reg::StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, preg_n_b16);
        }
    }
    AscendC::Reg::StoreUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb),
                                                                              ureg_exp_sum, 0);
}

// update, 64 < originN <= 128
template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128>
__aicore__ inline void ProcessVec1UpdateGeneralImpl128(const LocalTensor<T2> &dstTensor,
                                                       const LocalTensor<T> &srcTensor,
                                                       const LocalTensor<T> &inMaxTensor,
                                                       const LocalTensor<T> &sharedTmpBuffer, const uint16_t m,
                                                       const uint32_t originN, const T scale, const T minValue)
{
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    const uint32_t oriTailN = originN - floatRepSize;
    const uint32_t tailN = s2BaseSize - floatRepSize;
    uint32_t pltOriTailN = oriTailN;
    uint32_t pltTailN = tailN;
    uint32_t pltN = s2BaseSize;

    __ubuf__ T2 *expUb = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *inMaxUb = (__ubuf__ T *)inMaxTensor.GetPhyAddr();
    __ubuf__ T *tmpExpSumUb = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T *tmpMaxUb = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T *tmpMaxUb2 = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr() + 64;

    ProcessVec1UpdateGeneralImpl128VF<T, T2, s1BaseSize, s2BaseSize>(expUb, srcUb, inMaxUb, tmpExpSumUb, tmpMaxUb,
                                                                     tmpMaxUb2, blockStride, repeatStride, m, scale,
                                                                     minValue, pltOriTailN, pltTailN, pltN);
}
} // namespace SCFaVectorApi

#endif // VF_BASIC_BLOCK_UNALIGNED128_UPDATE_SCFA_H
