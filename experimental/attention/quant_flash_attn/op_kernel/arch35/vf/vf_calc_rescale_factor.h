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
 * \file vf_calc_rescale_factor.h
 * \brief V1 跨 S2 分块 rescale 因子（原 VfExpSub 改名）：整数减法 + IEEE fp32 位构造（无 exp），[128]
 */

#ifndef VF_CALC_RESCALE_FACTOR_H_
#define VF_CALC_RESCALE_FACTOR_H_

#include "kernel_tensor.h"
#include "vf_common_def_mxfp8_softmax_fp16.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

/*
 * @ingroup VfaVectorApi
 * @brief 计算跨 S2 分块 rescale 因子 softmaxExpUB[loop%3] = 2^(K^{blkStart} - K^{final})
 *
 * 输入均为 half log2 整数域量化指数（整数 k）。差值为整数，通过 IEEE fp32 位构造
 * 直接拼出 2^ΔK 的位模式（指数域 = ΔK + 127，尾数域全零），精确无舍入。
 * V2 侧（VfFlashUpdateAcc）按 [DV,S1] 元素级段加载使用。
 *
 * @param [out] softmaxExp   输出 [128] fp32（softmaxExpUB 槽位，IEEE 位模式）
 * @param [in]  blkStartMax  块开始量化指数 K^{blkStart}（blockStartMaxUB，half 整数）
 * @param [in]  finalMax     块最终量化指数 K^{final}（softmaxMaxUB[stateSlot]，half 整数）
 *
 * 指令链：Sub(half 整数减) → Adds 127 → Maxs 0
 *        → Cast<int32> ×2（RegLayout ZERO/ONE 各取偶/奇位元素，宽度展开 half[128]→int32[64]×2）
 *        → ShiftLefts 23（移入 fp32 指数域）→ Interleave（偶奇重组成顺序）→ Store ×2
 *
 * RegLayout 说明：h2iZero/h2iOne 的舍入模式相同（CAST_ROUND），区别在 RegLayout 字段
 * （ZERO=取偶数位元素，ONE=取奇数位元素）——控制宽度变化时的元素路由，非舍入模式。
 */
__simd_vf__ inline void VfCalcRescaleFactorVF(__ubuf__ float* softmaxExp, __ubuf__ half* blkStartMax,
                                              __ubuf__ half* finalMax)
{
    RegTensor<half> vreg_blk_start;
    RegTensor<half> vreg_final;
    RegTensor<half> vreg_delta_k; // ΔK = K^{blkStart} - K^{final}（整数，≤ 0）
    RegTensor<float> vreg_bits_a; // fp32 位模式（声明 float，Cast/ShiftLefts 时强转 int32——MxFP4 同款）
    RegTensor<float> vreg_bits_b; // fp32 位模式（同上）
    MaskReg preg_half = CreateMask<half, MaskPattern::ALL>();
    MaskReg preg_i32 = CreateMask<int32_t, MaskPattern::ALL>();

    // [128] half 输入 = 1 个寄存器（无循环，一次走完——half 域 1 寄存器装 128 元素；
    // 原 fp32 方案需 2 轮循环是每寄存器仅 64 元素之故，域切换后循环已无意义）
    // ① half 域整数运算：ΔK = K^{blkStart} - K^{final}
    LoadAlign(vreg_blk_start, blkStartMax);
    LoadAlign(vreg_final, finalMax);
    Sub(vreg_delta_k, vreg_blk_start, vreg_final, preg_half);

    // ② +127（fp32 指数偏置）并防御负值
    Adds(vreg_delta_k, vreg_delta_k, NUM_127, preg_half);
    Maxs(vreg_delta_k, vreg_delta_k, ZERO_VALUE, preg_half);

    // ③ 宽度展开：half[128] → int32[64]×2
    //    RegLayout::ZERO 取偶数位元素（e0, e2, ...），RegLayout::ONE 取奇数位（e1, e3, ...）
    //    （两次 Cast 各填一个 256B int32 寄存器，舍入模式相同，非双舍入）
    Cast<int32_t, half, h2iZero>((RegTensor<int32_t>&)vreg_bits_a, vreg_delta_k, preg_half);
    Cast<int32_t, half, h2iOne>((RegTensor<int32_t>&)vreg_bits_b, vreg_delta_k, preg_half);

    // ④ ShiftLefts 23：整数值移入 fp32 指数域位置
    //    (ΔK + 127) << 23 = fp32 位模式，尾数全零，指数域 = ΔK + 127
    ShiftLefts((RegTensor<int32_t>&)vreg_bits_a, (RegTensor<int32_t>&)vreg_bits_a, SHIFT_VALUE, preg_i32);
    ShiftLefts((RegTensor<int32_t>&)vreg_bits_b, (RegTensor<int32_t>&)vreg_bits_b, SHIFT_VALUE, preg_i32);

    // ⑤ Interleave：偶奇交错重组成顺序（A=前64，B=后64），存为 fp32
    Interleave((RegTensor<int32_t>&)vreg_bits_a, (RegTensor<int32_t>&)vreg_bits_b, (RegTensor<int32_t>&)vreg_bits_a,
               (RegTensor<int32_t>&)vreg_bits_b);
    StoreAlign(softmaxExp, vreg_bits_a, preg_i32);      // 输出 [0..63]
    StoreAlign(softmaxExp + 64, vreg_bits_b, preg_i32); // 输出 [64..127]
}

/*
 * @ingroup VfaVectorApi
 * @brief VfCalcRescaleFactor 的 __aicore__ wrapper：接收 LocalTensor + 槽位索引，内部转指针调 vf
 *
 * @param [in] softmaxExpUB     rescale 因子输出（float 视图，[taskSlot][128] 布局）
 * @param [in] blockStartMaxUB  块首快照 m_{k-1}（[128]）
 * @param [in] softmaxMaxUB     accMax 缓冲（[stateSlot][128] 布局）
 * @param [in] taskSlot         任务槽（0..2）
 * @param [in] stateSlot        accMax 状态槽（0..2）
 */
__aicore__ inline void VfCalcRescaleFactor(const LocalTensor<float>& softmaxExpUB,
                                           const LocalTensor<half>& blockStartMaxUB,
                                           const LocalTensor<half>& softmaxMaxUB, uint32_t taskSlot, uint32_t stateSlot)
{
    constexpr uint32_t smaxSlot = 128; // 单槽 half 元素数
    __ubuf__ float* expPtr = reinterpret_cast<__ubuf__ float*>(softmaxExpUB.GetPhyAddr()) + taskSlot * smaxSlot;
    __ubuf__ half* blkStartPtr = reinterpret_cast<__ubuf__ half*>(blockStartMaxUB.GetPhyAddr());
    __ubuf__ half* finalMaxPtr = reinterpret_cast<__ubuf__ half*>(softmaxMaxUB.GetPhyAddr()) + stateSlot * smaxSlot;
    VfCalcRescaleFactorVF(expPtr, blkStartPtr, finalMaxPtr);
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_CALC_RESCALE_FACTOR_H_
