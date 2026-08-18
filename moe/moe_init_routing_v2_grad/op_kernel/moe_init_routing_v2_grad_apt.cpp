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
 * \file moe_init_routing_v2_grad_apt.cpp
 * \brief
 */
#include "arch35/moe_init_routing_v2_grad_full_load.h"
#include "arch35/moe_init_routing_v2_grad_split_h.h"
#include "arch35/moe_init_routing_v2_grad_arch35.h"

using namespace AscendC;
using namespace MoeInitRoutingV2Grad;

#define TILINGKEY_SPLIT_H_DROPLESS 200001
#define TILINGKEY_SPLIT_H_DROP_PAD 200002
#define TILINGKEY_SPLIT_H_ACTIVE 200003
#define TILINGKEY_FULL_LOAD_DROPLESS 300001
#define TILINGKEY_FULL_LOAD_DROP_PAD 300002
#define TILINGKEY_FULL_LOAD_ACTIVE 300003
#define TILINGKEY_REGBASE_DROPLESS 400001
#define TILINGKEY_REGBASE_DROP_PAD 400002
#define TILINGKEY_REGBASE_ACTIVE 400003

extern "C" __global__ __aicore__ void moe_init_routing_v2_grad(
    GM_ADDR gradExpandedX,
    GM_ADDR expandedRowIdx,
    GM_ADDR gradX,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipeOp;
    if (TILING_KEY_IS(TILINGKEY_SPLIT_H_DROPLESS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseSplitHTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradSplitHCompute<DTYPE_GRAD_EXPANDED_X> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_SPLIT_H_DROP_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseSplitHTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradSplitHCompute<DTYPE_GRAD_EXPANDED_X, DROP_PAD_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_SPLIT_H_ACTIVE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseSplitHTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradSplitHCompute<DTYPE_GRAD_EXPANDED_X, ACTIVE_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_FULL_LOAD_DROPLESS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseFullLoadTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseFullLoadTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradFullLoadCompute<DTYPE_GRAD_EXPANDED_X> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_FULL_LOAD_DROP_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseFullLoadTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseFullLoadTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradFullLoadCompute<DTYPE_GRAD_EXPANDED_X, DROP_PAD_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_FULL_LOAD_ACTIVE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseFullLoadTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseFullLoadTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradFullLoadCompute<DTYPE_GRAD_EXPANDED_X, ACTIVE_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_REGBASE_DROPLESS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradCompute<DTYPE_GRAD_EXPANDED_X> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_REGBASE_DROP_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradCompute<DTYPE_GRAD_EXPANDED_X, DROP_PAD_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_REGBASE_ACTIVE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradCompute<DTYPE_GRAD_EXPANDED_X, ACTIVE_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    }

    pipeOp.Destroy();
}
