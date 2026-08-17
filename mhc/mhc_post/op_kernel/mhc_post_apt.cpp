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
 * \file mhc_post_apt.cpp
 * \brief MhcPost kernel entry point
 * Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
 *          nohres: x_{l+1} = x_l + h_{l}^{out} * H_{t}^{post}
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/mhc_post_arch35.h"
#include "arch35/mhc_post_regbase.h"
#include "arch35/mhc_post_nohres.h"

using namespace AscendC;
using namespace MhcPost;

template <uint16_t USE_PERMANENT_X, uint16_t USE_REGBASE, uint16_t USE_H_RES>
__global__ __aicore__ void mhc_post(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost, GM_ADDR output,
                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MhcPostRegbaseTilingData);
    TPipe tPipe;

    if constexpr (USE_H_RES == 0) {
        REGISTER_TILING_FOR_TILINGKEY("USE_H_RES == 0", MhcPostRegbaseTilingData);
        GET_TILING_DATA_WITH_STRUCT(MhcPostRegbaseTilingData, tilingData, tiling);
        MhcPostNoHRes<DTYPE_X> op(&tPipe, &tilingData);
        op.Init(x, hOut, hPost, output, workspace);
        op.Process();
    } else if constexpr (USE_REGBASE == 0) {
        REGISTER_TILING_FOR_TILINGKEY("USE_REGBASE == 0", MhcPostTilingData);
        GET_TILING_DATA_WITH_STRUCT(MhcPostTilingData, tilingData, tiling);
        MhcPostKernel<DTYPE_X, USE_PERMANENT_X> op(&tPipe, &tilingData);
        op.Init(x, hRes, hOut, hPost, output, workspace);
        op.Process();
    } else {
        REGISTER_TILING_FOR_TILINGKEY("USE_REGBASE == 1", MhcPostRegbaseTilingData);
        GET_TILING_DATA_WITH_STRUCT(MhcPostRegbaseTilingData, tilingData, tiling);
        MhcPostRegbase<DTYPE_X> op(&tPipe, &tilingData);
        op.Init(x, hRes, hOut, hPost, output, workspace);
        op.Process();
    }
}
