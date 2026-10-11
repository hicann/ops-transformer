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
 * \file moe_gating_top_k.cpp
 * \brief
 */

#include "moe_gating_top_k_e_k_fullload.h"
#include "moe_gating_top_k_without_group.h"
#include "moe_gating_top_k_generalized.h"

#define TILING_KEY_PER_GROUP_COUNT_32 0
#define TILING_KEY_WITHOUT_GROUP 1
#define TILING_KEY_GENERALIZED 2
#define TILING_KEY_HASH_WITHOUT_GROUP_INT32_INT64 3
#define TILING_KEY_HASH_WITHOUT_GROUP_INT32_INT32 4
#define TILING_KEY_HASH_WITHOUT_GROUP_INT64_INT64 5
#define TILING_KEY_HASH_WITHOUT_GROUP_INT64_INT32 6

using namespace AscendC;
using namespace MoeGatingTopK;
extern "C" __global__ __aicore__ void moe_gating_top_k(GM_ADDR x, GM_ADDR bias, GM_ADDR inputIds, GM_ADDR tid2eid,
                                                       GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKTilingData, tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    const MoeGatingTopKTilingData* __restrict t = &tilingData;
    TPipe tPipe;
    if (TILING_KEY_IS(TILING_KEY_PER_GROUP_COUNT_32)) {
        MoeGatingTopKEKFullload<DTYPE_X> op;
        op.Init(x, bias, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_GROUP)) {
        MoeGatingTopKWithoutGroup<DTYPE_X> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GENERALIZED)) {
        MoeGatingTopKGenerlized<DTYPE_X> op;
        op.Init(x, bias, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HASH_WITHOUT_GROUP_INT32_INT64)) {
        MoeGatingTopKWithoutGroup<DTYPE_X, int32_t, int64_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HASH_WITHOUT_GROUP_INT32_INT32)) {
        MoeGatingTopKWithoutGroup<DTYPE_X, int32_t, int32_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HASH_WITHOUT_GROUP_INT64_INT64)) {
        MoeGatingTopKWithoutGroup<DTYPE_X, int64_t, int64_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HASH_WITHOUT_GROUP_INT64_INT32)) {
        MoeGatingTopKWithoutGroup<DTYPE_X, int64_t, int32_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    }
}
