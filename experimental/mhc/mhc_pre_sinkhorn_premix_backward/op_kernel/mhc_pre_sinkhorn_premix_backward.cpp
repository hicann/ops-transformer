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
 * \file mhc_pre_sinkhorn_premix_backward.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch22/mhc_pre_grad_kernel.h"
#include "arch22/mhc_pre_sinkhorn_premix_backward_data_arch22.h"
#include "arch22/mhc_pre_sinkhorn_premix_backward_key_arch22.h"

using namespace AscendC;

template <bool TILINGKEY>
__global__ __aicore__ void mhc_pre_sinkhorn_premix_backward(GM_ADDR grad_hin, GM_ADDR grad_h_post, GM_ADDR grad_h_res,
                                                            GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                                            GM_ADDR h_pre, GM_ADDR hc_before_norm, GM_ADDR inv_rms,
                                                            GM_ADDR sum_out, GM_ADDR norm_out, GM_ADDR pre_mix,
                                                            GM_ADDR grad_pre, GM_ADDR grad_x, GM_ADDR grad_phi,
                                                            GM_ADDR grad_alpha, GM_ADDR grad_bias, GM_ADDR grad_pre_mix,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MhcPreSinkhornPremixBackwardArch22TilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornPremixBackwardArch22TilingData, tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    if constexpr (TILINGKEY) {
        MhcPreGradKernel<DTYPE_X, DTYPE_PHI, true> op;
        op.mm1_.SetSubBlockIdx(0);
        op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
        op.mm2_.SetSubBlockIdx(0);
        op.mm2_.Init(&tilingData.mm2TilingData, &pipe);

        op.Init(x, phi, h_pre, grad_hin, grad_h_post, grad_h_res, alpha, bias, hc_before_norm, inv_rms, sum_out,
                norm_out, grad_x, grad_phi, grad_alpha, grad_bias, pre_mix, grad_pre, grad_pre_mix, usrWorkspace,
                &tilingData, &pipe);
        op.Process();
    } else {
        MhcPreGradKernel<DTYPE_X, DTYPE_PHI, false> op;
        op.mm1_.SetSubBlockIdx(0);
        op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
        op.mm2_.SetSubBlockIdx(0);
        op.mm2_.Init(&tilingData.mm2TilingData, &pipe);

        op.Init(x, phi, h_pre, grad_hin, grad_h_post, grad_h_res, alpha, bias, hc_before_norm, inv_rms, sum_out,
                norm_out, grad_x, grad_phi, grad_alpha, grad_bias, pre_mix, grad_pre, grad_pre_mix, usrWorkspace,
                &tilingData, &pipe);
        op.Process();
    }
}
