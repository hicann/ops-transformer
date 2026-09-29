/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_calc_lse.h
 * \brief V2 末块 softmaxLse 输出计算：lse[i] = k_final[i]·ln2 + ln(rowsum[i])，[128] fp32
 */

#ifndef VF_CALC_LSE_H_
#define VF_CALC_LSE_H_

#include "kernel_tensor.h"
#include "vf_common_def_mxfp8_softmax_fp16.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

// ln(2) 的最近 fp32 值（golden 侧 double ln2 → f32 舍入同值域，误差 ≪ Atol）
constexpr float LN2_F32 = 0.6931471824645996f;

/*
 * @ingroup QfaVectorApi
 * @brief softmaxLse 计算（k 域整数指数 → ln 域）
 *
 * lse[i] = float(k_final[i]) * ln2 + Log(rowsum[i])——与 golden 逐位同式
 * （cpu_mxfp8_golden 末行：m_acc.float() * LN2 + log(s_safe)）。k 为 V1 在线
 * softmax 的 log2 整数域 accMax（softmaxMaxUB[stateSlot]），rowsum 为 V2 跨块
 * 累积分母终值（accRowsumUB）——P 已按 2^(k_i-k_final) 网格缩放，故 lse 由
 * k_final 域重建而非原始 max 域。
 *
 * 注：全宽 [128] 处理，[dealCount, 128) 为垃圾值但不拷出（GM 拷贝按有效行数截断）
 */
__simd_vf__ inline void VfCalcLseVF(__ubuf__ float* dst, __ubuf__ half* srcK, __ubuf__ float* srcSum)
{
    RegTensor<half> vreg_k;
    RegTensor<float> vreg_kf_a;  // k 展宽 fp32[64]（偶位）
    RegTensor<float> vreg_kf_b;  // k 展宽 fp32[64]（奇位）
    RegTensor<float> vreg_sum_a; // rowsum fp32[0..63]
    RegTensor<float> vreg_sum_b; // rowsum fp32[64..127]
    MaskReg preg_half = CreateMask<half, MaskPattern::ALL>();
    MaskReg preg_f32 = CreateMask<float, MaskPattern::ALL>();

    // ① k：half[128] 单寄存器载入 → Zero/One 宽度展开 → Interleave 重组为顺序 fp32[128]
    LoadAlign(vreg_k, srcK);
    Cast<float, half, h2iZero>(vreg_kf_a, vreg_k, preg_half);
    Cast<float, half, h2iOne>(vreg_kf_b, vreg_k, preg_half);
    Interleave((RegTensor<int32_t>&)vreg_kf_a, (RegTensor<int32_t>&)vreg_kf_b, (RegTensor<int32_t>&)vreg_kf_a,
               (RegTensor<int32_t>&)vreg_kf_b);

    // ② rowsum：fp32[128] 两寄存器载入
    LoadAlign(vreg_sum_a, srcSum);
    LoadAlign(vreg_sum_b, srcSum + 64);

    // ③ lse = k·ln2 + ln(rowsum)
    Muls(vreg_kf_a, vreg_kf_a, LN2_F32, preg_f32);
    Muls(vreg_kf_b, vreg_kf_b, LN2_F32, preg_f32);
    Log<float, MaskMergeMode::ZEROING>(vreg_sum_a, vreg_sum_a, preg_f32);
    Log<float, MaskMergeMode::ZEROING>(vreg_sum_b, vreg_sum_b, preg_f32);
    Add<float, MaskMergeMode::ZEROING>(vreg_kf_a, vreg_kf_a, vreg_sum_a, preg_f32);
    Add<float, MaskMergeMode::ZEROING>(vreg_kf_b, vreg_kf_b, vreg_sum_b, preg_f32);

    StoreAlign(dst, vreg_kf_a, preg_f32);
    StoreAlign(dst + 64, vreg_kf_b, preg_f32);
}

/*
 * @ingroup QfaVectorApi
 * @brief VfCalcLse 的 __aicore__ wrapper：接收 LocalTensor + 状态槽索引
 *
 * @param [out] lseUb          输出 [128] fp32（专用 LSE 暂存槽）
 * @param [in]  softmaxMaxUB   accMax 缓冲（[stateSlot][128] half 布局，k_final 终值）
 * @param [in]  accRowsumUB    V2 跨块累积分母（[128] fp32 终值）
 * @param [in]  stateSlot      accMax 状态槽（mloop % 3）
 */
__aicore__ inline void VfCalcLse(const LocalTensor<float>& lseUb, const LocalTensor<half>& softmaxMaxUB,
                                 const LocalTensor<float>& accRowsumUB, uint32_t stateSlot)
{
    constexpr uint32_t smaxSlot = 128;
    __ubuf__ float* dstPtr = reinterpret_cast<__ubuf__ float*>(lseUb.GetPhyAddr());
    __ubuf__ half* kPtr = reinterpret_cast<__ubuf__ half*>(softmaxMaxUB.GetPhyAddr()) + stateSlot * smaxSlot;
    __ubuf__ float* sumPtr = reinterpret_cast<__ubuf__ float*>(accRowsumUB.GetPhyAddr());
    VfCalcLseVF(dstPtr, kPtr, sumPtr);
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_CALC_LSE_H_
