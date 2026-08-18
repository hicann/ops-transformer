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
 * \file moe_finalize_routing_v2_grad_apt.cpp
 * \brief
 */
#include "arch35/moe_finalize_routing_v2_grad_without_scale_cut_h_arch35.h"
#include "arch35/moe_finalize_routing_v2_grad_without_scale_not_cut_h_arch35.h"
#include "arch35/moe_finalize_routing_v2_grad_regbase_not_cut_h.h"
#include "arch35/moe_finalize_routing_v2_grad_regbase_cut_h.h"

#define TILING_KEY_WITHOUT_SCALE_NOT_CUT_H 10001
#define TILING_KEY_WITHOUT_SCALE_CUT_H 10002
#define TILING_KEY_WITH_SCALE_NOT_CUT_H_WITHOUT_BIAS 20011
#define TILING_KEY_WITH_SCALE_NOT_CUT_H_WITH_BIAS 20021
#define TILING_KEY_WITH_SCALE_CUT_H_WITHOUT_BIAS 20012
#define TILING_KEY_WITH_SCALE_CUT_H_WITH_BIAS 20022

#ifndef DTYPE_SCALES
#define DTYPE_SCALES float
#endif

using namespace MoeFinalizeRoutingV2Grad;

extern "C" __global__ __aicore__ void moe_finalize_routing_v2_grad(
    GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR bias,
    GM_ADDR gradExpandedX, GM_ADDR gradScales, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_WITHOUT_SCALE_NOT_CUT_H)) {
        GET_TILING_DATA_WITH_STRUCT(MoeFinalizeRoutingV2GradTilingData, tiling_data_in, tiling);
        const MoeFinalizeRoutingV2GradTilingData* __restrict tilingData = &tiling_data_in;
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithoutScaleNotCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, gradExpandedX, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_SCALE_CUT_H)) {
        GET_TILING_DATA_WITH_STRUCT(MoeFinalizeRoutingV2GradTilingData, tiling_data_in, tiling);
        const MoeFinalizeRoutingV2GradTilingData* __restrict tilingData = &tiling_data_in;
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradWithoutScaleCutH<DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX> op;
        op.Init(gradY, expandedRowIdx, gradExpandedX, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITH_SCALE_NOT_CUT_H_WITH_BIAS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeFinalizeRoutingV2GradNotSplitHTilingData, tiling_data_in, tiling);
        const MoeFinalizeRoutingV2GradNotSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradRegbaseNotCutH<
            DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX, DTYPE_SCALES, true>
            op;
        op.Init(
            gradY, expandedRowIdx, expandedX, scales, expertIdx, bias, gradExpandedX, gradScales, workspace, tilingData,
            &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITH_SCALE_NOT_CUT_H_WITHOUT_BIAS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeFinalizeRoutingV2GradNotSplitHTilingData, tiling_data_in, tiling);
        const MoeFinalizeRoutingV2GradNotSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradRegbaseNotCutH<
            DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX, DTYPE_SCALES, false>
            op;
        op.Init(
            gradY, expandedRowIdx, expandedX, scales, expertIdx, bias, gradExpandedX, gradScales, workspace, tilingData,
            &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITH_SCALE_CUT_H_WITH_BIAS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeFinalizeRoutingV2GradSplitHTilingData, tiling_data_in, tiling);
        const MoeFinalizeRoutingV2GradSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradRegBaseCutH<
            DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX, DTYPE_SCALES, true>
            op;
        op.Init(
            gradY, expandedRowIdx, expandedX, scales, expertIdx, bias, gradExpandedX, gradScales, workspace, tilingData,
            &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITH_SCALE_CUT_H_WITHOUT_BIAS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeFinalizeRoutingV2GradSplitHTilingData, tiling_data_in, tiling);
        const MoeFinalizeRoutingV2GradSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradRegBaseCutH<
            DTYPE_GRAD_Y, DTYPE_EXPANDED_ROW_IDX, DTYPE_SCALES, false>
            op;
        op.Init(
            gradY, expandedRowIdx, expandedX, scales, expertIdx, bias, gradExpandedX, gradScales, workspace, tilingData,
            &pipe);
        op.Process();
    } else {
        return;
    }
}