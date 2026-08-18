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
 * \file moe_init_routing_quant_v2_apt.cpp
 * \brief
 */
#include "arch35/moe_v2_sort_one_core_quant_arch35.h"
#include "arch35/moe_v2_sort_multi_core_quant_arch35.h"
#include "arch35/moe_v2_mrgsort_out_quant_arch35.h"
#include "arch35/moe_v2_mrgsort_quant_arch35.h"

#include "arch35/moe_v2_expert_token_out_simt_quant_v2.h"
#include "arch35/moe_v2_expert_token_out_regbase.h"
#include "arch35/moe_v2_src_to_dst_op_simt_quant_arch35.h"
#include "arch35/moe_v2_src_to_dst_with_capacity_simt_quant_v2.h"
#include "arch35/moe_v2_gather_quant_simt.h"
#include "arch35/moe_v2_gather_dynamic_quant_droppad.h"
#include "arch35/moe_v2_gather_dynamic_quant_dropless.h"

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
    bool histWithRegBase = t->histWithRegBase == 1;

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
            if (histWithRegBase) {
                TPipe expertTokenRegOutPipe;
                MoeV2ExpertTokenOutRegBase expertTokenRegOutOp;
                expertTokenRegOutOp.Init<MoeInitRoutingQuantV2TilingData>(
                    expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t,
                    &expertTokenRegOutPipe);
                expertTokenRegOutOp.Process();
                expertTokenRegOutPipe.Destroy();

                TPipe expertTokenOutPipe;
                MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
                expertTokenOutOpSimt.Init<MoeInitRoutingQuantV2TilingData>(
                    expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t,
                    &expertTokenOutPipe);
                expertTokenOutOpSimt.Process<false>();
                expertTokenOutPipe.Destroy();
            } else {
                TPipe expertTokenOutPipe;
                MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
                expertTokenOutOpSimt.Init<MoeInitRoutingQuantV2TilingData>(
                    expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t,
                    &expertTokenOutPipe);
                expertTokenOutOpSimt.Process<true>();
                expertTokenOutPipe.Destroy();
            }
        }
        MoeV2SrcToDstOpSimt srcToDstOpSimt;
        srcToDstOpSimt.Init<MoeInitRoutingQuantV2TilingData>(expandedRowIdx, userWS, t);
        srcToDstOpSimt.Process();

    } else if (TILING_KEY_IS(10100) || TILING_KEY_IS(10110) || TILING_KEY_IS(11100) || TILING_KEY_IS(11110)) {
        if (histWithRegBase) {
            TPipe expertTokenRegOutPipe;
            MoeV2ExpertTokenOutRegBase expertTokenRegOutOp;
            expertTokenRegOutOp.Init<MoeInitRoutingQuantV2TilingData>(
                expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t,
                &expertTokenRegOutPipe);
            expertTokenRegOutOp.Process();
            expertTokenRegOutPipe.Destroy();

            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
            expertTokenOutOpSimt.Init<MoeInitRoutingQuantV2TilingData>(
                expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t, &expertTokenOutPipe);
            expertTokenOutOpSimt.Process<false>();
            expertTokenOutPipe.Destroy();
        } else {
            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
            expertTokenOutOpSimt.Init<MoeInitRoutingQuantV2TilingData>(
                expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, userWS, t, &expertTokenOutPipe);
            expertTokenOutOpSimt.Process<true>();
            expertTokenOutPipe.Destroy();
        }

        MoeV2SrcToDstWithCapacitySimt<int8_t, MoeInitRoutingQuantV2TilingData> srcToDstWithCapacityOpSimt;
        srcToDstWithCapacityOpSimt.Init(expandedRowIdx, expandedX, userWS, t);
        srcToDstWithCapacityOpSimt.Process();
    }

    if (TILING_KEY_IS(10000) || TILING_KEY_IS(10010) || TILING_KEY_IS(10100) || TILING_KEY_IS(10110)) {
        TPipe gatherPipe;
        MoeV2GatherQuant<DTYPE_X> gatherQuantOp;
        gatherQuantOp.Init(x, scale, offset, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
        gatherQuantOp.Process();
        gatherPipe.Destroy();
    } else if (TILING_KEY_IS(11000) || TILING_KEY_IS(11010)) {
        TPipe gatherPipe;
        MoeV2GatherDynamicQuantDropless<DTYPE_X> gatherDynamicQuantDroplessOp;
        gatherDynamicQuantDroplessOp.Init(
            x, scale, expandedRowIdx, expandedX, dynamicQuantScale, userWS, t, &gatherPipe);
        gatherDynamicQuantDroplessOp.Process();
        gatherPipe.Destroy();
    } else if (TILING_KEY_IS(11100) || TILING_KEY_IS(11110)) {
        TPipe gatherPipe;
        MoeV2GatherDynamicQuantDroppad<DTYPE_X> gatherDynamicQuantDroppadOp;
        gatherDynamicQuantDroppadOp.Init(
            x, scale, expandedRowIdx, expandedX, dynamicQuantScale, userWS, t, &gatherPipe);
        gatherDynamicQuantDroppadOp.Process();
        gatherPipe.Destroy();
    }
}
