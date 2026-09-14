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
 * \file mhc_pre_backward.cpp
 * \brief
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/mhc_pre_backward_kernel.h"
#include "arch35/mhc_pre_backward_tiling_key.h"

using namespace AscendC;
using namespace MhcPreBackward;

template <int8_t TILING_MODE>
__global__ __aicore__ void mhc_pre_backward(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR gradHIn, GM_ADDR gradHPost,
                                            GM_ADDR gradHRes, GM_ADDR invRms, GM_ADDR hMix, GM_ADDR hPre, GM_ADDR hPost,
                                            GM_ADDR gamma, GM_ADDR gradXPostOptional, GM_ADDR gradX, GM_ADDR gradPhi,
                                            GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR gradGamma, GM_ADDR workspaceGM,
                                            GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    if constexpr (TILING_MODE == MHC_PRE_BACKWARD_DEFAULT) {
        InitParams initParams{x,         phi,   alpha, gradHIn,           gradHPost, gradHRes, invRms,    hMix,
                              hPre,      hPost, gamma, gradXPostOptional, gradX,     gradPhi,  gradAlpha, gradBias,
                              gradGamma, user,  &pipe, &tilingData};
        MhcPreBackwardKernel<DTYPE_X, float32_t> op;
        op.Init(initParams);
        op.Process();
    }
}
#endif
