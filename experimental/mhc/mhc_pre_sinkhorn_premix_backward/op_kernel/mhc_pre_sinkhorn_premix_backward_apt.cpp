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
 * \file mhc_pre_sinkhorn_premix_backward_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/mhc_pre_sinkhorn_premix_backward_one_stage.h"
#include "arch35/mhc_pre_sinkhorn_premix_backward_data_arch35.h"
#include "arch35/mhc_pre_sinkhorn_premix_backward_key_arch35.h"
#include "arch35/mhc_pre_sinkhorn_premix_backward_deterministic.h"

using namespace AscendC;
using namespace Reg;

template <bool DETERMINISTIC>
__global__ __aicore__ void mhc_pre_sinkhorn_premix_backward(GM_ADDR grad_hin, GM_ADDR grad_h_post, GM_ADDR grad_h_res,
                                                            GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                                            GM_ADDR h_pre, GM_ADDR hc_before_norm, GM_ADDR inv_rms,
                                                            GM_ADDR sum_out, GM_ADDR norm_out, GM_ADDR pre_mix,
                                                            GM_ADDR grad_pre, GM_ADDR grad_x, GM_ADDR grad_phi,
                                                            GM_ADDR grad_alpha, GM_ADDR grad_bias, GM_ADDR grad_pre_mix,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(MhcPreSinkhornPremixBackwardArch35TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    if constexpr (DETERMINISTIC) {
        REGISTER_TILING_FOR_TILINGKEY("DETERMINISTIC == true",
                                      MhcPreSinkhornPremixBackwardArch35DeterminiticTilingData);
        GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornPremixBackwardArch35DeterminiticTilingData, tilingData, tiling);
        MhcPreSinkhornPremixBackwardDeterministic<DTYPE_X, DTYPE_GRAD_HIN, float> op;
        op.mm1_.SetSubBlockIdx(0);
        op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
        op.mm2_.SetSubBlockIdx(0);
        op.mm2_.Init(&tilingData.mm2TilingData, &pipe);
        op.Init(x, phi, h_pre, grad_hin, grad_h_post, grad_h_res, alpha, bias, hc_before_norm, inv_rms, sum_out,
                norm_out, grad_x, grad_phi, grad_alpha, grad_bias, usrWorkspace, &tilingData, &pipe);
        op.Process();
    } else {
        REGISTER_TILING_FOR_TILINGKEY("DETERMINISTIC == false", MhcPreSinkhornPremixBackwardArch35TilingData);
        GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornPremixBackwardArch35TilingData, tilingData, tiling);
        MhcPreSinkhornPremixBackwardOneStage<DTYPE_PHI, DTYPE_GRAD_HIN> op;
        op.mm1_.SetSubBlockIdx(0);
        op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
        op.mm2_.SetSubBlockIdx(0);
        op.mm2_.Init(&tilingData.mm2TilingData, &pipe);
        op.Init(x, phi, h_pre, grad_hin, grad_h_post, grad_h_res, alpha, bias, hc_before_norm, inv_rms, sum_out,
                norm_out, grad_x, grad_phi, grad_alpha, grad_bias, usrWorkspace, &tilingData, &pipe);
        op.Process();
    }
}
