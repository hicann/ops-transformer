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
 * \file moe_init_routing_quant_v2.cpp
 * \brief
 */
#include "inner/moe_v2_sort_one_core_inner.h"
#include "inner/moe_v2_sort_multi_core_inner.h"
#include "inner/moe_v2_mrgsort_out_inner.h"
#include "inner/moe_v2_mrgsort_inner.h"
#include "inner/moe_v2_expert_token_out_inner.h"
#include "inner/moe_v2_src_to_dst_op_inner.h"
#include "inner/moe_v2_src_to_dst_with_capacity_inner.h"
#include "moe_v2_fullload_quant.h"
#include "moe_v2_fullload_dynamic_quant.h"
#include "moe_v2_gather_quant.h"
#include "moe_v2_gather_dynamic_quant.h"
#include "moe_v2_src_to_dst_and_gather.h"

using namespace AscendC;
using namespace MoeInitRoutingQuantV2;
extern "C" __global__ __aicore__ void moe_init_routing_quant_v2(
    GM_ADDR x,
    GM_ADDR expertIdx,
    GM_ADDR scale,
    GM_ADDR offset,
    GM_ADDR expandedX,
    GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum,
    GM_ADDR expertTokensBeforeCapacity,
    GM_ADDR dynamicQuantScale,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);  
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    auto t = &tilingData;
    if (TILING_KEY_IS(20000)) { // quant full load
        TPipe sortPipe;
        MoeV2FullLoadQuant<DTYPE_X> op;
        op.Init(
            x, expertIdx, scale, offset, expandedX, expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &sortPipe);
        op.Process();
        sortPipe.Destroy();
        return;
    } else if (TILING_KEY_IS(21000)) { // dynamic quant full load
        TPipe sortPipe;
        MoeV2FullLoadDynamicQuant<DTYPE_X, DTYPE_EXPANDED_X> op;
        op.Init(
            x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, scale, dynamicQuantScale, userWS, t,
            &sortPipe);
        op.Process();
        sortPipe.Destroy();
        return;
    }

    // sort
    if (TILING_KEY_IS(10000) || TILING_KEY_IS(10100) || TILING_KEY_IS(11000) || TILING_KEY_IS(11100)) {
        TPipe sortPipe;
        MoeV2SortOneCore op;
        op.Init<MoeInitRoutingQuantV2TilingData>(
            expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else if (TILING_KEY_IS(10010) || TILING_KEY_IS(10110) || TILING_KEY_IS(11010) || TILING_KEY_IS(11110)) {
        TPipe sortPipe;
        MoeV2SortMultiCore op;
        op.Init<MoeInitRoutingQuantV2TilingData>(
            expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    }

    if (TILING_KEY_IS(10000) || TILING_KEY_IS(10010) || TILING_KEY_IS(11000) || TILING_KEY_IS(11010)) {
        if (t->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOut expertTokenOutOp;
            expertTokenOutOp.Init<MoeInitRoutingQuantV2TilingData>(
                expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t, &expertTokenOutPipe);
            expertTokenOutOp.Process();
            expertTokenOutPipe.Destroy();
        }
        TPipe srcToDstPipe;
        MoeV2SrcToDstOp srcToDstOp;
        srcToDstOp.Init<MoeInitRoutingQuantV2TilingData>(expandedRowIdx, userWS, t, &srcToDstPipe);
        srcToDstOp.Process();
        srcToDstPipe.Destroy();
    } else if (TILING_KEY_IS(10100) || TILING_KEY_IS(10110) || TILING_KEY_IS(11100) || TILING_KEY_IS(11110)) {
        TPipe expertTokenOutPipe;
        MoeV2ExpertTokenOut expertTokenOutOp;
        expertTokenOutOp.Init<MoeInitRoutingQuantV2TilingData>(
            expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t, &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();

        if (TILING_KEY_IS(10100) || TILING_KEY_IS(10110)) {
            TPipe srcToDstPipe;
            MoeV2SrcToDstWithCapacity<int8_t, MoeInitRoutingQuantV2TilingData> srcToDstWithCapacityOp;
            srcToDstWithCapacityOp.Init(expandedRowIdx, expandedX, userWS, t, &srcToDstPipe);
            srcToDstWithCapacityOp.Process();
            srcToDstPipe.Destroy();
        } else {
            TPipe srcToDstGatherPipe;
            MoeV2SrcToDstAndGather<DTYPE_X, MoeInitRoutingQuantV2TilingData> srcToDstAndGatherOp;
            srcToDstAndGatherOp.Init(
                x, scale, expandedRowIdx, expandedX, dynamicQuantScale, userWS, t, &srcToDstGatherPipe);
            srcToDstAndGatherOp.Process();
            srcToDstGatherPipe.Destroy();
            return;
        }
    }

    if (TILING_KEY_IS(10000) || TILING_KEY_IS(10010) || TILING_KEY_IS(10100) || TILING_KEY_IS(10110)) {
        TPipe gatherPipe;
        MoeV2GatherQuant<DTYPE_X> gatherQuantOp;
        gatherQuantOp.Init(x, scale, offset, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
        gatherQuantOp.Process();
        gatherPipe.Destroy();
    } else if (TILING_KEY_IS(11000) || TILING_KEY_IS(11010)) {
        TPipe gatherPipe;
        MoeV2GatherDynamicQuant<DTYPE_X, DTYPE_EXPANDED_X> gatherDynamicQuantOp;
        gatherDynamicQuantOp.Init(x, scale, expandedRowIdx, expandedX, dynamicQuantScale, userWS, t, &gatherPipe);
        gatherDynamicQuantOp.Process();
        gatherPipe.Destroy();
    }
}
