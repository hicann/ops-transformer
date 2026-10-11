/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_finalize_routing_v2_grad.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_grad_with_scale_cut_h.h"
#include "moe_finalize_routing_v2_grad_with_scale_not_cut_h.h"
#include "moe_finalize_routing_v2_grad_without_scale_cut_h.h"
#include "moe_finalize_routing_v2_grad_without_scale_not_cut_h.h"

#define TILING_KEY_WITHOUT_SCALE_NOT_CUT_H 10001
#define TILING_KEY_WITHOUT_SCALE_CUT_H 10002
#define TILING_KEY_WITH_SCALE_NOT_CUT_H 20001
#define TILING_KEY_WITH_SCALE_CUT_H 20002

using namespace MoeFinalizeRoutingV2Grad;

extern "C" __global__ __aicore__ void moe_finalize_routing_v2_grad(GM_ADDR gradY, GM_ADDR expandedRowIdx,
                                                                   GM_ADDR expandedX, GM_ADDR scales, GM_ADDR expertIdx,
                                                                   GM_ADDR bias, GM_ADDR gradExpandedX,
                                                                   GM_ADDR gradScales, GM_ADDR workspace,
                                                                   GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);

    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_WITHOUT_SCALE_NOT_CUT_H)) {
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithoutScaleNotCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, gradExpandedX, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_SCALE_CUT_H)) {
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithoutScaleCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, gradExpandedX, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITH_SCALE_NOT_CUT_H)) {
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithScaleNotCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, expandedX, scales, expertIdx, bias, gradExpandedX, gradScales, workspace,
                &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITH_SCALE_CUT_H)) {
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithScaleCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, expandedX, scales, expertIdx, bias, gradExpandedX, gradScales, workspace,
                &tilingData, &pipe);
        op.Process();
    } else {
        return;
    }
}
