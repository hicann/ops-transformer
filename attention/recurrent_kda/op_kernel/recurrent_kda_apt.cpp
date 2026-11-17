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
 * \file recurrent_kda_apt.cpp
 * \brief
 */
#include "arch35/recurrent_kda_arch35.h"
#include "recurrent_kda_apt_tiling_key.h"
#include "recurrent_kda_tiling_data.h"

using namespace AscendC;
using namespace RecurrentKda;

template <uint32_t KERNEL_MODE>
__global__ __aicore__ void recurrent_kda(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gate, GM_ADDR beta,
                                         GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR aLog,
                                         GM_ADDR dtBias, GM_ADDR numAcceptedTokens, GM_ADDR out,
                                         GM_ADDR initialStateOut, GM_ADDR finalState, GM_ADDR workspaceGM,
                                         GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(RecurrentKdaTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    InitSocState();
    GM_ADDR stateOutput = tilingData.inplaceFinalState == 1 ? initialStateOut : finalState;
    constexpr bool stateVFirst = KERNEL_MODE == RKDA_TPL_V_FIRST || KERNEL_MODE == RKDA_TPL_V_FIRST_FUSED;
    constexpr bool fusedMode = KERNEL_MODE == RKDA_TPL_V_FIRST_FUSED || KERNEL_MODE == RKDA_TPL_K_FIRST_FUSED;
    RecurrentKdaStand::RecurrentKdaEngine<DTYPE_INITIAL_STATE, DTYPE_GATE, DTYPE_BETA, stateVFirst, fusedMode> op(
        &tilingData);
    op.Init(query, key, value, gate, beta, initialState, cuSeqlens, ssmStateIndices, aLog, dtBias, numAcceptedTokens,
            out, stateOutput);
    op.Process();
}
