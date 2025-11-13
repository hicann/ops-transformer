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
 * \file moe_distribute_combine_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_v2.h"

#if __has_include("../moe_distribute_combine/moe_distribute_combine_a2.h")
#include "../moe_distribute_combine/moe_distribute_combine_a2.h"
#include "../moe_distribute_combine/moe_distribute_combine_a2_layered.h"
#include "../moe_distribute_combine/moe_distribute_combine_a2_layered_aicpu.h"
#else
#include "../../moe_distribute_combine/op_kernel/moe_distribute_combine_a2.h"
#include "../../moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h"
#include "../../moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered_aicpu.h"
#endif

using namespace MoeDistributeCombineV2Impl;
using namespace MoeDistributeCombineA2Impl;
using namespace AscendC;

namespace {
template <TemplateMC2TypeClass>
__aicore__ inline void ExecMoeDistributeCombineV2(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                                GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR scales,
                                                GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
                                                GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                                GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM,
                                                GM_ADDR tilingGM, TPipe *pipePtr)
{
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    MoeDistributeCombineV2<TemplateMC2TypeFunc> op;
    op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask,
        sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, pipePtr, &tilingData);
    op.Process();
}
}

/*
* A3 tilingkey说明
* 5位的十进制数
* 第1位（个位）：无意义占位使用
* 第2位（十位）：通信量化选项：
*     0：无量化, 2:int8量化
* 第3位（百位）：是否做tp域allgather:
*     0: 不做, 1: 做
* 第4位（千位）：无实际意义:
* 第5位（万位）：无实际意义.
*/

extern "C" __global__ __aicore__ void moe_distribute_combine_v2(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                                             GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                             GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                             GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
                                                             GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                                             GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut,
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 10000", MoeDistributeCombineA2TilingData);
    TPipe pipe;

#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(10100)) { // tp=2 IsInt8Quant=0
        ExecMoeDistributeCombineV2<DTYPE_EXPAND_X, DTYPE_X, int32_t, true, false>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(10000)) { // tp=1 IsInt8Quant=0
        ExecMoeDistributeCombineV2<DTYPE_EXPAND_X, DTYPE_X, int32_t, false, false>(expandX, expertIds, assistInfoForCombine, 
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(10120)) { // tp=2 IsInt8Quant=1
        ExecMoeDistributeCombineV2<DTYPE_EXPAND_X, DTYPE_X, int32_t, true, true>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(10020)) { // tp=1 IsInt8Quant=1
        ExecMoeDistributeCombineV2<DTYPE_EXPAND_X, DTYPE_X, int32_t, false, true>(expandX, expertIds, assistInfoForCombine,
            epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(2000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
        MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
        op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, scales, xActiveMask, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(3000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeCombineA2LayeredAicpu<DTYPE_EXPAND_X, int32_t> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, expandScales, XOut, workspaceGM, &pipe, &tilingData,
                contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t, DTYPE_EXPAND_X> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, expandScales, XOut, workspaceGM, &pipe, &tilingData,
                contextGM0);
            op.Process();
        }
    }
    if (TILING_KEY_IS(3100)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t, int8_t> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, expandScales, XOut, workspaceGM, &pipe, &tilingData,
                contextGM0);
            op.Process();
        }
    }
#endif
}