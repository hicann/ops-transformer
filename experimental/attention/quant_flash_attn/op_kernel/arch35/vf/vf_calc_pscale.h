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
 * \file vf_calc_pscale.h
 * \brief V1 末 subLoop 统一算 pscale：Δk+127 → e8m0 字节对，直落 768B 网格影像槽
 */

#ifndef VF_CALC_PSCALE_H_
#define VF_CALC_PSCALE_H_

#include "kernel_tensor.h"
#include "vf_common_def_mxfp8_softmax_fp16.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

/*
 * @ingroup VfaVectorApi
 * @brief 计算第 i 个 subLoop 的 pscale = e8m0(k_i - K_final + 127)，直落 768B 网格影像槽
 *
 * 输入均为 half log2 整数域量化指数。输出字节对影像 [v,v,v,v,…]（e8m0 值重复对，
 * 与 oneFill E8M0_ONE_U16=0x7F7F 同构——L1 Mx 网格单元内每列 2 字节的契约）。
 * 网格契约（subLoop 粒度槽）：单元 (x, y) 地址 = (3x+y)×32B，x=0..7 为 S1 16 列组
 * （× 2 字节对），y=0..1 为 S2 64 行组 + 1 bank pad（读侧 yStart 恒 0、srcStride=3）。
 * subLoop 的值广播到 y=0/1 两行（槽即本 subLoop 专属，无跨 subLoop 偏移）。
 * 末 subLoop 的 k_i = K_final（accMax 单调累积）→ Δk 恒 0 → 127，通用路径无特判。
 *
 * @param [out] pscaleGrid  768B 网格影像槽（pscaleGridUB[ring]，uint8 视图）
 * @param [in]  subLoopMax  本 subLoop 量化指数 k_i（subLoopUsedMaxUB[i]，half 整数）
 * @param [in]  finalMax    分块最终量化指数 K_final（softmaxMaxUB[stateSlot]，half 整数）
 */
__simd_vf__ inline void VfCalcPScaleVF(__ubuf__ uint8_t* pscaleGrid, __ubuf__ half* subLoopMax, __ubuf__ half* finalMax)
{
    RegTensor<half> vreg_sub_loop_max;
    RegTensor<half> vreg_final_max;
    RegTensor<half> vreg_delta_k;       // Δk = k_i - K_final（整数，≤ 0；末 subLoop 恒 0）
    RegTensor<uint8_t> vreg_scale_even; // Cast 偶位产物（奇位清零）
    RegTensor<uint8_t> vreg_scale_odd;  // Cast 奇位产物（偶位清零）
    MaskReg preg_half = CreateMask<half, MaskPattern::ALL>();
    MaskReg preg_u8 = CreateMask<uint8_t, MaskPattern::ALL>();

    // ① half 域整数运算：Δk = k_i - K_final，+127 → e8m0 字节值（负值钳 0 = 2^-127）
    LoadAlign(vreg_sub_loop_max, subLoopMax);
    LoadAlign(vreg_final_max, finalMax);
    Sub(vreg_delta_k, vreg_sub_loop_max, vreg_final_max, preg_half);
    Adds(vreg_delta_k, vreg_delta_k, NUM_127, preg_half);
    Maxs(vreg_delta_k, vreg_delta_k, ZERO_VALUE, preg_half);

    // ② 字节对影像：Cast Zero（偶位）/ One（奇位）同源配对 + Or → [v,v,v,v,…] 256B
    //    （8 个 32B 块，块 x = S1 列组 x 的 16 列字节对——单元内 2 字节/列契约）
    Cast<uint8_t, half, castTraitZero>(vreg_scale_even, vreg_delta_k, preg_half);
    Cast<uint8_t, half, castTraitOne>(vreg_scale_odd, vreg_delta_k, preg_half);
    Or(vreg_scale_even, vreg_scale_even, vreg_scale_odd, preg_u8);

    // ③ 散布到网格 y=0 / y=1 两行（同值广播）：256B 寄存器 = 8 个 32B 块，
    //    blockStride=3（32B 单位）→ 块 x 落 base + x×96B = (3x+y)×32B
    StoreAlign<uint8_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(pscaleGrid + 0U, vreg_scale_even, 3U, preg_u8);
    StoreAlign<uint8_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(pscaleGrid + 32U, vreg_scale_even, 3U, preg_u8);
}

/*
 * @ingroup VfaVectorApi
 * @brief VfCalcPScale 的 __aicore__ wrapper：接收 LocalTensor + 槽位索引，内部转指针调 vf
 *
 * @param [in] pscaleGridUB  768B 网格影像缓冲（[ring][768B] 布局，uint8 视图）
 * @param [in] usedMaxUB     subLoop k_i 缓冲（[subLoopIdx][128] half 布局）
 * @param [in] softmaxMaxUB  accMax 缓冲（[stateSlot][128] half 布局）
 * @param [in] ringIdx       ring 槽号（(loop×2+subLoopIdx) mod 6）
 * @param [in] subLoopIdx    subLoop 序号（0..1，k_i 槽索引）
 * @param [in] stateSlot     accMax 状态槽（0..2）
 */
__aicore__ inline void VfCalcPScale(const LocalTensor<uint8_t>& pscaleGridUB, const LocalTensor<half>& usedMaxUB,
                                    const LocalTensor<half>& softmaxMaxUB, uint32_t ringIdx, uint32_t subLoopIdx,
                                    uint32_t stateSlot)
{
    __ubuf__ uint8_t* gridPtr =
        reinterpret_cast<__ubuf__ uint8_t*>(pscaleGridUB.GetPhyAddr()) + ringIdx * QFA_UB_PSCALE_SLOT;
    __ubuf__ half* subLoopMaxPtr =
        reinterpret_cast<__ubuf__ half*>(usedMaxUB.GetPhyAddr()) + subLoopIdx * QFA_UB_SMAX_SLOT;
    __ubuf__ half* finalMaxPtr =
        reinterpret_cast<__ubuf__ half*>(softmaxMaxUB.GetPhyAddr()) + stateSlot * QFA_UB_SMAX_SLOT;
    VfCalcPScaleVF(gridPtr, subLoopMaxPtr, finalMaxPtr);
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_CALC_PSCALE_H_
