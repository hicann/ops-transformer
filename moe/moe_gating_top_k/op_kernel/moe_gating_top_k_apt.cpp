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
 * \file moe_gating_top_k_apt.cpp
 * \brief
 */

#include "arch35/moe_gating_top_k_regbase.h"
#include "arch35/moe_gating_top_k_without_group_regbase.h"
#include "arch35/moe_gating_top_k_e_k_fullload_regbase.h"
using namespace AscendC;
using namespace MoeGatingTopK;

#define TILING_KEY_REGBASE 10000
#define TILING_KEY_HASH_INT32_INT64 10001
#define TILING_KEY_HASH_INT32_INT32 10002
#define TILING_KEY_HASH_INT64_INT64 10003
#define TILING_KEY_HASH_INT64_INT32 10004
#define TILING_KEY_WITHOUT_GROUP_REGBASE 10005
#define TILING_KEY_E_K_FULLLOAD_REGBASE 10010

extern "C" __global__ __aicore__ void moe_gating_top_k(GM_ADDR x, GM_ADDR bias, GM_ADDR inputIds, GM_ADDR tid2eid,
                                                       GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKRegbaseTilingData, tiling_data_in, tiling);
    const MoeGatingTopKRegbaseTilingData *__restrict tilingData = &tiling_data_in;
    TPipe tPipe;

    if (TILING_KEY_IS(TILING_KEY_E_K_FULLLOAD_REGBASE)) {
        MoeGatingTopKEKFullloadRegbase<DTYPE_X> op;
        op.Init(x, bias, y, expertIdx, out, userWS, tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_GROUP_REGBASE)) {
        MoeGatingTopKWithoutGroupRegbase<DTYPE_X> op;
        op.Init(x, bias, y, expertIdx, out, userWS, tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_REGBASE) || TILING_KEY_IS(TILING_KEY_HASH_INT32_INT32)) {
        MoeGatingTopKRegbase<DTYPE_X> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HASH_INT32_INT64)) {
        MoeGatingTopKRegbase<DTYPE_X, int32_t, int64_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HASH_INT64_INT64)) {
        MoeGatingTopKRegbase<DTYPE_X, int64_t, int64_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HASH_INT64_INT32)) {
        MoeGatingTopKRegbase<DTYPE_X, int64_t, int32_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, tilingData, &tPipe);
        op.Process();
    }
}
