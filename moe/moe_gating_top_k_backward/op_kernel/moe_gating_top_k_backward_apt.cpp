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
 * \file moe_gating_top_k_backward_apt.cpp
 * \brief
 */

#include "./arch35/moe_gating_top_k_backward.h"
#include "./arch35/moe_gating_top_k_backward_struct.h"
using namespace AscendC;
using namespace MoeGatingTopKBackwardNs;

#define TILING_KEY_REGBASE 10000

extern "C" __global__ __aicore__ void moe_gating_top_k_backward(GM_ADDR xNorm, GM_ADDR gradY, GM_ADDR expertIdx,
                                                                GM_ADDR gradX, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }

    REGISTER_TILING_DEFAULT(MoeGatingTopKBackwardA5TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKBackwardA5TilingData, tilingData, tiling);
    const MoeGatingTopKBackwardA5TilingData *__restrict t = &tilingData;
    TPipe tPipe;

    if (TILING_KEY_IS(TILING_KEY_REGBASE)) {
        MoeGatingTopKBackward<DTYPE_GRAD_Y> op;
        op.Init(xNorm, gradY, expertIdx, gradX, t, &tPipe);
        op.Process();
    }
}
