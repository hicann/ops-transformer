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
 * \file moe_init_routing.cpp
 * \brief
 */
#include "moe_gather_out.h"
#include "moe_mrgsort_out.h"
#include "moe_mrgsort.h"
#include "moe_sort_multi_core.h"
#include "moe_sort_one_core.h"
#include "moe_src_to_dst_op.h"
#include "moe_gather_out.h"
#include "moe_gather_out_small_activate_row.h"
#include "moe_init_routing_fullload.h"

using namespace AscendC;
using namespace MoeInitRouting;
extern "C" __global__ __aicore__ void moe_init_routing(GM_ADDR x, GM_ADDR rowIdx, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                       GM_ADDR expandedRowIdx, GM_ADDR expandedExpertIdx,
                                                       GM_ADDR workspace, GM_ADDR tiling)
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

    if (TILING_KEY_IS(0)) {
        TPipe sortPipe;
        MoeFullLoad<DTYPE_X> op;
        op.Init(x, rowIdx, expertIdx, expandedX, expandedRowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        op.Process();
        return;
    }

    TPipe sortPipe;
    // sort
    if (TILING_KEY_IS(1)) {
        MoeSortOneCore op;
        op.Init(expertIdx, rowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        MoeSortOneCore op;
        op.Init(expertIdx, rowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        MoeSortMultiCore op;
        op.Init(expertIdx, rowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        MoeSortMultiCore op;
        op.Init(expertIdx, rowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        op.Process();
    }
    sortPipe.Destroy();

    TPipe srcToDstPipe;
    MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init(expandedRowIdx, userWS, t, &srcToDstPipe);
    srcToDstOp.Process();
    srcToDstPipe.Destroy();

    TPipe gatherPipe;
    if (TILING_KEY_IS(1)) {
        MoeGatherOut<DTYPE_X> gatherOp;
        gatherOp.Init(x, expandedRowIdx, expandedX, t, &gatherPipe);
        gatherOp.Process();
    } else if (TILING_KEY_IS(2)) {
        MoeGatherOut<DTYPE_X> gatherOp;
        gatherOp.Init(x, expandedRowIdx, expandedX, t, &gatherPipe);
        gatherOp.Process();
    } else if (TILING_KEY_IS(3)) {
        MoeGatherOutSmallActiveRow<DTYPE_X> gatherOp;
        gatherOp.Init(x, userWS, expandedRowIdx, expandedX, t, &gatherPipe);
        gatherOp.Process();
    } else if (TILING_KEY_IS(4)) {
        MoeGatherOutSmallActiveRow<DTYPE_X> gatherOp;
        gatherOp.Init(x, userWS, expandedRowIdx, expandedX, t, &gatherPipe);
        gatherOp.Process();
    }
    gatherPipe.Destroy();
}
