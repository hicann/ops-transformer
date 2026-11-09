/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "text_gated_residual_grad_constants.h"

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == HC_GRAD_CCE_A5
#define CATLASS_ARCH HC_GRAD_ARCH_A5
#else
#define CATLASS_ARCH HC_GRAD_ARCH_A2
#endif
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/text_gated_residual_grad_tiling_data_arch35.h"
#include "arch35/text_gated_residual_grad_tiling_key_arch35.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == HC_GRAD_CCE_A5
#include "arch35/text_gated_residual_grad_kernel_arch35.h"
#else
// The v220 MIX wrapper calls matmul::clearWorkspace; include its declaration only. CATLASS handles computation.
#include "lib/matmul_intf.h"
#include "arch22/text_gated_residual_grad_kernel_arch22.h"
#endif

using namespace AscendC;
using namespace TextGatedResidualGrad;

template <typename D_T, int HAS_GAMMA>
__global__ __aicore__ void text_gated_residual_grad(GM_ADDR grad_mixed, GM_ADDR x, GM_ADDR w_down, GM_ADDR w_up,
                                                    GM_ADDR inv_rms, GM_ADDR down_pre_act, GM_ADDR up,
                                                    GM_ADDR grad_inject, GM_ADDR w_gate, GM_ADDR gamma, GM_ADDR inject,
                                                    GM_ADDR dx, GM_ADDR dw_down, GM_ADDR dw_up, GM_ADDR dw_gate,
                                                    GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(TextGatedResidualGradTilingDataArch35);
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == HC_GRAD_CCE_A5
    // CATLASS Arch::Resource owns the Cube TPipe. Construct the outer TPipe only for AIV
    // to avoid event-resource conflicts with older CATLASS versions.
    if ASCEND_IS_AIV {
        TPipe tPipe;
        KernelTextGatedResidualGradArch35<D_T, HAS_GAMMA> op;
        op.Init(grad_mixed, x, w_down, w_up, inv_rms, down_pre_act, up, grad_inject, w_gate, gamma, inject, dx, dw_down,
                dw_up, dw_gate, dgamma, userWS, &tilingData, &tPipe);
        op.Process();
        tPipe.Destroy();
    } else {
        KernelTextGatedResidualGradArch35<D_T, HAS_GAMMA> op;
        op.Init(grad_mixed, x, w_down, w_up, inv_rms, down_pre_act, up, grad_inject, w_gate, gamma, inject, dx, dw_down,
                dw_up, dw_gate, dgamma, userWS, &tilingData, nullptr);
        op.Process();
    }
#else
    // On A2/A3, CATLASS Arch::Resource owns the Cube TPipe. A second outer TPipe would compete
    // for the default TPipe and can hang GEMM after L1/L0 initialization.
    // Only AIV Vector execution needs the outer TPipe; construct it according to the core type.
    if ASCEND_IS_AIV {
        TPipe tPipe;
        KernelTextGatedResidualGradArch22<D_T, HAS_GAMMA> op;
        op.Init(grad_mixed, x, w_down, w_up, inv_rms, down_pre_act, up, grad_inject, w_gate, gamma, inject, dx, dw_down,
                dw_up, dw_gate, dgamma, userWS, &tilingData, &tPipe);
        op.Process();
        tPipe.Destroy();
    } else {
        KernelTextGatedResidualGradArch22<D_T, HAS_GAMMA> op;
        op.Init(grad_mixed, x, w_down, w_up, inv_rms, down_pre_act, up, grad_inject, w_gate, gamma, inject, dx, dw_down,
                dw_up, dw_gate, dgamma, userWS, &tilingData, nullptr);
        op.Process();
    }
#endif
}
