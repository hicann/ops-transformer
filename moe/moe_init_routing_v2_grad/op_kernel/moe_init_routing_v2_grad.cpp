/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v2_grad.cpp
 * \brief
 */
#include "moe_init_routing_v2_grad_with_dropless.h"
#include "moe_init_routing_v2_grad_with_activate.h"
#include "moe_init_routing_v2_grad_with_pos_drop_and_pad_zero.h"
/// test
using namespace AscendC;
using namespace MoeInitRoutingV2Grad;

#define MOE_INIT_ROUTING_V2_GRAD_DROPLESS_FLOAT 1000UL
#define MOE_INIT_ROUTING_V2_GRAD_DROPLESS_FLOAT16 1001UL
#define MOE_INIT_ROUTING_V2_GRAD_DROPLESS_BF16 1002UL
#define MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_FLOAT 1010UL
#define MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_FLOAT16 1011UL
#define MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_BF16 1012UL
#define MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_FLOAT 1100UL
#define MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_FLOAT16 1101UL
#define MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_BF16 1102UL

extern "C" __global__ __aicore__ void moe_init_routing_v2_grad(GM_ADDR gradExpandedX, GM_ADDR expandedRowIdx,
                                                               GM_ADDR gradX, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    TPipe pipeOp;
    if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_DROPLESS_FLOAT)) {
        MoeInitRoutingV2GradDroplessCompute<DTYPE_GRAD_EXPANDED_X> op;
        op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_DROPLESS_FLOAT16)) {
        MoeInitRoutingV2GradDroplessCompute<DTYPE_GRAD_EXPANDED_X> op;
        op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_DROPLESS_BF16)) {
        MoeInitRoutingV2GradDroplessCompute<DTYPE_GRAD_EXPANDED_X> op;
        op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_FLOAT)) {
        MoeInitRoutingV2GradActivateCompute<DTYPE_GRAD_EXPANDED_X> op;
        op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_FLOAT16)) {
        MoeInitRoutingV2GradActivateCompute<DTYPE_GRAD_EXPANDED_X> op;
        op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_ACTIVATE_BF16)) {
        MoeInitRoutingV2GradActivateCompute<DTYPE_GRAD_EXPANDED_X> op;
        op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_FLOAT)) {
        MoeInitRoutingV2GradPositionPad0Compute<DTYPE_GRAD_EXPANDED_X> op;
        op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_FLOAT16)) {
        MoeInitRoutingV2GradPositionPad0Compute<DTYPE_GRAD_EXPANDED_X> op;
        op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V2_GRAD_POSITION_PAD0_BF16)) {
        MoeInitRoutingV2GradPositionPad0Compute<DTYPE_GRAD_EXPANDED_X> op;
        op.Init(gradExpandedX, expandedRowIdx, gradX, &tilingData, &pipeOp);
        op.Process();
    }
    pipeOp.Destroy();
}
