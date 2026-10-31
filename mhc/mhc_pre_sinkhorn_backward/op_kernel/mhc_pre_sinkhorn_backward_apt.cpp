/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Ascend960 uses dedicated tiling data and keys.
#if defined(__DAV_920R1__) || defined(__DAV_920R2__) || \
    (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
#include "arch92/mhc_pre_sinkhorn_backward_entry_arch92.h"
#else
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/mhc_pre_sinkhorn_backward_data_arch35.h"
#include "arch35/mhc_pre_sinkhorn_backward_key_arch35.h"
#include "arch35/mhc_pre_sinkhorn_backward_kernel_arch35.h"

using namespace AscendC;
using namespace MhcPreSinkhornBackwardNs;

// N is determined by runtime tiling data; different N values share the same kernel instance.
template <typename T, SplitMode MODE, bool DETERMINISTIC>
__aicore__ inline void MhcPreSinkhornBackwardRun(GM_ADDR grad_hin, GM_ADDR grad_h_post, GM_ADDR grad_h_res, GM_ADDR x,
                                                 GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR h_pre,
                                                 GM_ADDR hc_before_norm, GM_ADDR inv_rms, GM_ADDR sum_out,
                                                 GM_ADDR norm_out, GM_ADDR grad_x, GM_ADDR grad_phi, GM_ADDR grad_alpha,
                                                 GM_ADDR grad_bias, GM_ADDR userWorkspace,
                                                 const MhcPreSinkhornBackwardArch35TilingData* tilingData, TPipe* pipe)
{
    MhcPreSinkhornBackwardKernel<T, MODE, DETERMINISTIC> op;
    op.Init(grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
            grad_x, grad_phi, grad_alpha, grad_bias, userWorkspace, tilingData, pipe);
    op.Process();
}

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

    if (TILING_KEY_IS(3000)) {
        if (workspace == nullptr || tilingData->userWorkspaceBytes == 0) {
            return;
        }
        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        MhcPreSinkhornBackwardRun<DTYPE_X, SplitMode::MK_SPLIT, false>(
            grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
            grad_x, grad_phi, grad_alpha, grad_bias, userWorkspace, tilingData, &pipe);
    } else if (TILING_KEY_IS(3001)) {
        MhcPreSinkhornBackwardRun<DTYPE_X, SplitMode::M_SPLIT, false>(
            grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
            grad_x, grad_phi, grad_alpha, grad_bias, nullptr, tilingData, &pipe);
    } else if (TILING_KEY_IS(3003)) {
        // The deterministic branch shares the M_SPLIT instance.
        MhcPreSinkhornBackwardRun<DTYPE_X, SplitMode::M_SPLIT, false>(
            grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
            grad_x, grad_phi, grad_alpha, grad_bias, nullptr, tilingData, &pipe);
    }
}

#endif
