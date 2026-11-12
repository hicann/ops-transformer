/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Ascend960 使用独立的 Tiling 数据和 TilingKey。
#if defined(__DAV_920R1__) || defined(__DAV_920R2__) || \
    (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
#include "arch92/mhc_pre_sinkhorn_backward_entry_arch92.h"
#else
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/mhc_pre_sinkhorn_backward_data_arch35.h"
#include "arch35/mhc_pre_sinkhorn_backward_kernel_arch35.h"

using namespace AscendC;
using namespace MhcPreSinkhornBackwardNs;

// arch35 入口：在 MhcPreBackward 的 V0 阶段融合 Sinkhorn 反向。
extern "C" __global__ __aicore__ void mhc_pre_sinkhorn_backward(
    GM_ADDR grad_hin, GM_ADDR grad_h_post, GM_ADDR grad_h_res, GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
    GM_ADDR h_pre, GM_ADDR hc_before_norm, GM_ADDR inv_rms, GM_ADDR sum_out, GM_ADDR norm_out, GM_ADDR grad_x,
    GM_ADDR grad_phi, GM_ADDR grad_alpha, GM_ADDR grad_bias, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MhcPreSinkhornBackwardArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornBackwardArch35TilingData, tilingDataValue, tiling);
    const MhcPreSinkhornBackwardArch35TilingData* tilingData = &tilingDataValue;
    TPipe pipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    InitParams params;
    params.x = x;
    params.phi = phi;
    params.alpha = alpha;
    params.grad_h_in = grad_hin;
    params.grad_h_post = grad_h_post;
    params.grad_h_res = grad_h_res;
    params.inv_rms = inv_rms;
    params.h_mix = hc_before_norm;
    params.h_pre = h_pre;
    params.bias = bias;
    params.sum_out = sum_out;
    params.norm_out = norm_out;
    params.grad_x = grad_x;
    params.grad_phi = grad_phi;
    params.grad_alpha = grad_alpha;
    params.grad_bias = grad_bias;
    params.workspace = userWorkspace;
    params.tPipeIn = &pipe;
    params.tilingData = tilingData;

    // 大小 M 共用同一实例，Stage2 策略由 tilingData->smallM 控制。
    if (TILING_KEY_IS(3000)) {
        MhcPreSinkhornBackwardKernel<DTYPE_X, float32_t> op;
        op.Init(params);
        op.Process();
    }
}

#endif
