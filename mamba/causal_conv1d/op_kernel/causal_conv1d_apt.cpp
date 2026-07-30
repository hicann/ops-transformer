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
 * \file causal_conv1d_apt.cpp
 * \brief Kernel entry point — dispatches to FN or Update implementation.
 */

#include "arch35/causal_conv1d_fn.h"
#include "arch35/causal_conv1d_update.h"

namespace {

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
__aicore__ inline void RunCausalConv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR bias,
                                       GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                       GM_ADDR numAcceptedTokens, GM_ADDR convStatesOut, GM_ADDR y, GM_ADDR workspace,
                                       const CausalConv1dTilingData *tilingData)
{
    if constexpr (CausalConv1d::IsFnInputModeKey(inputModeKey)) {
        CausalConv1d::RunCausalConv1dFn<T, inputModeKey, widthKey, hasBiasKey, activationKey>(
            x, weight, convStates, bias, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
            convStatesOut, y, workspace, tilingData);
    } else {
        CausalConv1d::RunCausalConv1dUpdate<T, inputModeKey, widthKey, hasBiasKey, activationKey>(
            x, weight, convStates, bias, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
            convStatesOut, y, workspace, tilingData);
    }
}

} // namespace

template <uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
__global__ __aicore__ void causal_conv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR bias,
                                         GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                         GM_ADDR numAcceptedTokens, GM_ADDR convStatesOut, GM_ADDR y, GM_ADDR workspace,
                                         GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR userWorkspace = workspace;
    if (workspace != nullptr) {
        userWorkspace = AscendC::GetUserWorkspace(workspace);
    }

    RunCausalConv1d<DTYPE_X, inputModeKey, widthKey, hasBiasKey, activationKey>(
        x, weight, convStates, bias, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, convStatesOut, y,
        userWorkspace, &tilingData);
}
