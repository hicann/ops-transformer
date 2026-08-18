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
 * \file moe_init_routing.cpp
 * \brief
 */
#include "arch35/moe_gather_out_arch35.h"
#include "arch35/moe_mrgsort_out_arch35.h"
#include "arch35/moe_mrgsort_arch35.h"
#include "arch35/moe_sort_multi_core_arch35.h"
#include "arch35/moe_sort_one_core_arch35.h"
#include "arch35/moe_src_to_dst_op_arch35.h"
#include "arch35/moe_gather_out_arch35.h"
#include "arch35/moe_gather_out_small_activate_row_arch35.h"
#include "arch35/moe_init_routing_fullload_arch35.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/moe_src_to_dst_simt_op.h"
#endif

using namespace AscendC;
using namespace MoeInitRouting;
extern "C" __global__ __aicore__ void moe_init_routing(
    GM_ADDR x, GM_ADDR rowIdx, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR expandedExpertIdx,
    GM_ADDR workspace, GM_ADDR tiling)
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

    if (TILING_KEY_IS(0)) {
        TPipe sortPipe;
        MoeFullLoad<DTYPE_X> op;
        op.Init(x, rowIdx, expertIdx, expandedX, expandedRowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        op.Process();
        return;
    }

    TPipe sortPipe;
    // sort
    if (TILING_KEY_IS(1) || TILING_KEY_IS(3)) {
        MoeSortOneCore op;
        op.Init(expertIdx, rowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(2) || TILING_KEY_IS(4)) {
        MoeSortMultiCore op;
        op.Init(expertIdx, rowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        op.Process();
    }
    sortPipe.Destroy();

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    MoeSrcToDstSimtOp srcToDstSimtOp;
    srcToDstSimtOp.Init(expandedRowIdx, userWS, t);
    srcToDstSimtOp.Process();
#else
    TPipe srcToDstPipe;
    MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init(expandedRowIdx, userWS, t, &srcToDstPipe);
    srcToDstOp.Process();
    srcToDstPipe.Destroy();
#endif

    TPipe gatherPipe;
    if (TILING_KEY_IS(1) || TILING_KEY_IS(2)) {
        MoeGatherOut<DTYPE_X> gatherOp;
        gatherOp.Init(x, expandedRowIdx, expandedX, t, &gatherPipe);
        gatherOp.Process();
    } else if (TILING_KEY_IS(3) || TILING_KEY_IS(4)) {
        MoeGatherOutSmallActiveRow<DTYPE_X> gatherOp;
        gatherOp.Init(x, userWS, expandedRowIdx, expandedX, t, &gatherPipe);
        gatherOp.Process();
    }
    gatherPipe.Destroy();
}
