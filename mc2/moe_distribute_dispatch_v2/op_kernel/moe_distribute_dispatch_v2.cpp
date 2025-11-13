/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_dispatch_v2_full_mesh.h"

#if __has_include("../moe_distribute_dispatch/moe_distribute_dispatch_a2.h")
#include "../moe_distribute_dispatch/moe_distribute_dispatch_a2.h"
#include "../moe_distribute_dispatch/moe_distribute_dispatch_a2_layered.h"
#include "../moe_distribute_dispatch/moe_distribute_dispatch_a2_layered_aicpu.h"
#else
#include "../../moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2.h"
#include "../../moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2_layered.h"
#include "../../moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2_layered_aicpu.h"
#endif

#include "moe_distribute_dispatch_v2.h"

using namespace MoeDistributeDispatchV2Impl;
using namespace MoeDistributeDispatchA2Impl;
using namespace MoeDistributeDispatchV2FullMeshImpl;

using namespace AscendC;
/*
* A3 tilingkey说明
* 5位的十进制数
* 第1位（个位）：quantMode:
*     0: 不量化, 1: 静态量化, 2: 动态量化
* 第2位（十位）：是否有smoothScale:
*     0: 无, 1: 有
* 第3位（百位）：是否做tp域allgather:
*     0: 不做, 1: 做
* 第4位（千位）：是否走fullmesh_v2模板:
*     0: 不做, 1: 做
* 第5位（万位）：无实际意义
*/

extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 2000000000", MoeDistributeDispatchA2TilingData);

    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10100)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, false, false, true> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(11000)) { // fullMesh高性能模板
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(2000001000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, tilingGM);
        op.Process();
    }
    if (TILING_KEY_IS(2100001000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2LayeredAicpu<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        }
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(10011)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, true, false, false, false> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, true, false> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10111)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, true, false, false, true> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10102)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, false, true> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10112)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, true, true> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(11011)) { // fullMesh高性能模板
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, true, false, false, false> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(11002)) { // fullMesh高性能模板
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(11012)) { // fullMesh高性能模板
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, false, true, true, false> op;
        op.Init(x,expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(2000001002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, tilingGM);
        op.Process();
    }
    if (TILING_KEY_IS(2000001012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, tilingGM);
        op.Process();
    }
    if (TILING_KEY_IS(2100001002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2LayeredAicpu<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        }
    }
    if (TILING_KEY_IS(2100001012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2LayeredAicpu<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        }
    }
#endif
}