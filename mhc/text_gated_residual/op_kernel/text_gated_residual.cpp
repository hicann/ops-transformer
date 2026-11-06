/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
// v220 MIX 包装会调用 matmul::clearWorkspace。只引入声明，计算仍走 Catlass。
#include "lib/matmul_intf.h"
#endif

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/kernel/hc_forward_mul_sum_kernel.hpp"
#include "arch35/kernel/hc_small_fused_kernel.hpp"
#else
// CATLASS MatrixCoord uses unsigned indices but internally constructs int zero coordinates.
// Keep this upstream compatibility diagnostic local to CATLASS header definitions.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-narrowing"
#endif
#include "arch22/kernel/hc_forward_small_kernel.hpp"
#include "arch22/kernel/hc_forward_large_kernel.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

namespace {
__aicore__ inline bool HcValidGm(GM_ADDR p)
{
    return p != nullptr && reinterpret_cast<uint64_t>(p) != 0xffffffffffffffffULL;
}

template <class Element, uint32_t C1SwizzleDir, bool SmallN, class Tiling = HcForwardMulSumTilingData>
__aicore__ inline void RunHcForward(GM_ADDR x, GM_ADDR wDown, GM_ADDR wUp, GM_ADDR wGate, GM_ADDR gamma, GM_ADDR mixed,
                                    GM_ADDR inject, GM_ADDR invRms, GM_ADDR downPreAct, GM_ADDR upPreAct,
                                    GM_ADDR userWs, GM_ADDR tiling, const Tiling& tilingData)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    using Kernel = std::conditional_t<
        SmallN, Catlass::Gemm::Kernel::HcForwardSmallKernel<Element, C1SwizzleDir, Catlass::Arch::AtlasA2, Tiling>,
        Catlass::Gemm::Kernel::HcForwardLargeKernel<Element, C1SwizzleDir, Catlass::Arch::AtlasA2, Tiling>>;
#else
    static_assert(!SmallN, "A5 small shapes use HcSmallFused");
    using Kernel = Catlass::Gemm::Kernel::HcForwardMulSumKernel<Element, C1SwizzleDir, Tiling>;
#endif
    typename Kernel::Params params{};
    params.ptrX = x;
    params.ptrWDown = wDown;
    params.ptrWUp = wUp;
    params.ptrWGate = wGate;
    params.ptrGamma = gamma;
    params.ptrMixed = mixed;
    params.ptrInject = inject;
    const bool needBackward = (tilingData.needBackward == 1) && HcValidGm(invRms) && HcValidGm(downPreAct);
    params.ptrInvRms = needBackward ? invRms : nullptr;
    params.ptrDownPreAct = needBackward ? downPreAct : nullptr;
    params.ptrUpPreAct = needBackward && HcValidGm(upPreAct) ? upPreAct : nullptr;
    params.userWs = userWs;
    params.tiling = tiling;
    params.tilingFromUnpacked = 1;
    params.tilingUnpacked = &tilingData;
    Kernel{}(params);
}
} // namespace

extern "C" __global__ __aicore__ void text_gated_residual(GM_ADDR x, GM_ADDR w_down, GM_ADDR w_up, GM_ADDR w_gate,
                                                          GM_ADDR gamma, GM_ADDR mixed, GM_ADDR inject, GM_ADDR inv_rms,
                                                          GM_ADDR down_pre_act, GM_ADDR up_pre_act, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);

    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWs = AscendC::GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(TextGatedResidualTilingData, tilingData, tiling);

#if !defined(DTYPE_X)
    using Element = bfloat16_t;
#else
    using Element = DTYPE_X;
#endif
    static_assert(!std::is_same_v<Element, float>, "FP32 x is not supported");
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if (tilingData.M == 0u) {
        return;
    }
#endif
    if (TILING_KEY_IS(0)) {
        RunHcForward<Element, 0, false>(x, w_down, w_up, w_gate, gamma, mixed, inject, inv_rms, down_pre_act,
                                        up_pre_act, userWs, tiling, tilingData);
    } else if (TILING_KEY_IS(1)) {
        RunHcForward<Element, 1, false>(x, w_down, w_up, w_gate, gamma, mixed, inject, inv_rms, down_pre_act,
                                        up_pre_act, userWs, tiling, tilingData);
    } else if (TILING_KEY_IS(2)) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        const bool cache = tilingData.needBackward == 1u && HcValidGm(inv_rms) && HcValidGm(down_pre_act);
        Catlass::Gemm::Kernel::HcSmallFused<Element>(
            x, w_down, w_up, w_gate, gamma, mixed, inject, cache ? inv_rms : nullptr, cache ? down_pre_act : nullptr,
            cache && HcValidGm(up_pre_act) ? up_pre_act : nullptr, userWs, tilingData);
#else
        RunHcForward<Element, 0, true>(x, w_down, w_up, w_gate, gamma, mixed, inject, inv_rms, down_pre_act, up_pre_act,
                                       userWs, tiling, tilingData);
#endif
    } else if (TILING_KEY_IS(3)) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        const bool cache = tilingData.needBackward == 1u && HcValidGm(inv_rms) && HcValidGm(down_pre_act);
        Catlass::Gemm::Kernel::HcSmallFused<Element>(
            x, w_down, w_up, w_gate, gamma, mixed, inject, cache ? inv_rms : nullptr, cache ? down_pre_act : nullptr,
            cache && HcValidGm(up_pre_act) ? up_pre_act : nullptr, userWs, tilingData);
#else
        RunHcForward<Element, 1, true>(x, w_down, w_up, w_gate, gamma, mixed, inject, inv_rms, down_pre_act, up_pre_act,
                                       userWs, tiling, tilingData);
#endif
    }
}
