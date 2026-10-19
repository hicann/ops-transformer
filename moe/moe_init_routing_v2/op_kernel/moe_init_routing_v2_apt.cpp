/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_apt.cpp
 * \brief
 */

#include "arch35/moe_v2_mrgsort_out_arch35.h"
#include "arch35/moe_v2_mrgsort_arch35.h"
#include "arch35/moe_v2_sort_multi_core_arch35.h"
#include "arch35/moe_v2_sort_one_core_arch35.h"
#include "arch35/moe_v2_expert_token_out_regbase.h"
#include "arch35/moe_v2_expert_token_out_simt.h"
#include "arch35/moe_v2_src_to_dst_op_simt_arch35.h"
#include "arch35/moe_v2_src_to_dst_with_capacity_simt.h"
#include "arch35/moe_v2_gather_out_for_simt.h"
#include "arch35/moe_v2_scatter_out_batch_row.h"

using namespace AscendC;
using namespace MoeInitRoutingV2;

extern "C" __global__ __aicore__ void moe_init_routing_v2(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                          GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                          GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2Arch35TilingData, tilingData, tiling);
    using V2TilingData = MoeInitRoutingV2Arch35TilingData;
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    auto t = &tilingData;

    if (TILING_KEY_IS(10001) || TILING_KEY_IS(11001) || TILING_KEY_IS(10011) || TILING_KEY_IS(11011)) {
        TPipe sortPipe;
        MoeV2SortOneCore<DTYPE_EXPERT_IDX> op;
        op.Init(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else if (TILING_KEY_IS(10002) || TILING_KEY_IS(11002) || TILING_KEY_IS(10012) || TILING_KEY_IS(11012)) {
        TPipe sortPipe;
        MoeV2SortMultiCore<DTYPE_EXPERT_IDX> op;
        op.Init(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    }

    if (TILING_KEY_IS(11001) || TILING_KEY_IS(11002)) {
        if (t->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOutRegBase expertTokenOutOp;
            expertTokenOutOp.Init<V2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx,
                                                userWS, t, &expertTokenOutPipe);
            expertTokenOutOp.Process();
            expertTokenOutPipe.Destroy();
        }
    } else if (TILING_KEY_IS(11011) || TILING_KEY_IS(11012)) {
        TPipe expertTokenOutPipe;
        MoeV2ExpertTokenOutRegBase expertTokenOutOp;
        expertTokenOutOp.Init<V2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx,
                                            userWS, t, &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();

        TPipe expertTokenOutSimtPipe;
        MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
        expertTokenOutOpSimt.Init<V2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx,
                                                userWS, t, &expertTokenOutSimtPipe);
        expertTokenOutOpSimt.Process<false>();
        expertTokenOutSimtPipe.Destroy();
    } else if (TILING_KEY_IS(10001) || TILING_KEY_IS(10002)) {
        if (t->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
            expertTokenOutOpSimt.Init<V2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                    expandedRowIdx, userWS, t, &expertTokenOutPipe);
            expertTokenOutOpSimt.Process();
            expertTokenOutPipe.Destroy();
        }
    } else if (TILING_KEY_IS(10011) || TILING_KEY_IS(10012)) {
        TPipe expertTokenOutPipe;
        MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
        expertTokenOutOpSimt.Init<V2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx,
                                                userWS, t, &expertTokenOutPipe);
        expertTokenOutOpSimt.Process();
        expertTokenOutPipe.Destroy();
    }

    TPipe gatherPipe;
    bool isDroplessMode = TILING_KEY_IS(10001) || TILING_KEY_IS(11001) || TILING_KEY_IS(10002) || TILING_KEY_IS(11002);
    bool canParallel = t->gatherOutComputeParamsOp.scatterMode == 1;

    if (isDroplessMode) {
        MoeV2SrcToDstOpSimt srcToDstOpSimt;
        srcToDstOpSimt.Init<V2TilingData>(expandedRowIdx, userWS, t);
        if (canParallel) {
            srcToDstOpSimt.ProcessWithoutSync();
        } else {
            srcToDstOpSimt.Process();
        }
    } else {
        MoeV2SrcToDstWithCapacitySimt<DTYPE_X, V2TilingData> srcToDstWithCapacityOpSimt;
        srcToDstWithCapacityOpSimt.Init(expandedRowIdx, expandedX, userWS, t);
        if (canParallel) {
            srcToDstWithCapacityOpSimt.ProcessWithoutSync();
        } else {
            srcToDstWithCapacityOpSimt.Process();
        }
    }

    if (canParallel) {
        MoeV2ScatterOutBatchRow<DTYPE_X> scatterOpBatchRow;
        scatterOpBatchRow.Init(x, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
        scatterOpBatchRow.Process();
        AscendC::SyncAll();
    } else {
        MoeV2GatherOutSimt<DTYPE_X> gatherOpSimt;
        gatherOpSimt.Init(x, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
        gatherOpSimt.Process();
    }
    gatherPipe.Destroy();
}
