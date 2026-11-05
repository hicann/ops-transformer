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
 * \file attention_update_apt.cpp
 * \brief
 */
#include "arch35/attention_update_with_lse_regbase.h"
#include "arch35/attention_update_without_lse_regbase.h"

#define TILING_KEY_EMPTY 10000
#define TILING_KEY_UPDATE_TYPE_ZERO 20000
#define TILING_KEY_UPDATE_TYPE_ONE 20001
#define TILING_KEY_ROW_STRIDED_ZERO 21000
#define TILING_KEY_ROW_STRIDED_ONE 21001

using namespace AttentionUpdateOpt;

extern "C" __global__ __aicore__ void attention_update(GM_ADDR lse, GM_ADDR go, GM_ADDR out, GM_ADDR outLseMax,
                                                       GM_ADDR workSpace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(TILING_KEY_EMPTY)) {
        return;
    } else if (TILING_KEY_IS(TILING_KEY_ROW_STRIDED_ZERO)) {
        GET_TILING_DATA_WITH_STRUCT(AttentionUpdateStridedTilingData, data, tiling);
        AttentionUpdateTilingData baseData;
        baseData.sp = data.sp;
        baseData.d = data.d;
        baseData.usedCoreNum = data.usedCoreNum;
        baseData.perCoreCount = data.perCoreCount;
        baseData.lastCoreCount = data.lastCoreCount;
        baseData.perCoreLoops = data.perCoreLoops;
        baseData.lastCoreLoops = data.lastCoreLoops;
        baseData.perCorePerLoopCount = data.perCorePerLoopCount;
        baseData.perCoreLastLoopCount = data.perCoreLastLoopCount;
        baseData.lastCoreLastLoopCount = data.lastCoreLastLoopCount;
        baseData.bshInLoop = data.bshInLoop;

        AttentionUpdateWithoutLse<DTYPE_GO, true> op(&pipe, &baseData, data.rowStride);
        op.Init(lse, go, out, outLseMax, workSpace);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_ROW_STRIDED_ONE)) {
        GET_TILING_DATA_WITH_STRUCT(AttentionUpdateStridedTilingData, data, tiling);
        AttentionUpdateTilingData baseData;
        baseData.sp = data.sp;
        baseData.d = data.d;
        baseData.usedCoreNum = data.usedCoreNum;
        baseData.perCoreCount = data.perCoreCount;
        baseData.lastCoreCount = data.lastCoreCount;
        baseData.perCoreLoops = data.perCoreLoops;
        baseData.lastCoreLoops = data.lastCoreLoops;
        baseData.perCorePerLoopCount = data.perCorePerLoopCount;
        baseData.perCoreLastLoopCount = data.perCoreLastLoopCount;
        baseData.lastCoreLastLoopCount = data.lastCoreLastLoopCount;
        baseData.bshInLoop = data.bshInLoop;

        AttentionUpdateWithLse<DTYPE_GO, true> op(&pipe, &baseData, data.rowStride);
        op.Init(lse, go, out, outLseMax, workSpace);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UPDATE_TYPE_ZERO)) {
        GET_TILING_DATA_WITH_STRUCT(AttentionUpdateTilingData, tilingData, tiling);
        AttentionUpdateWithoutLse<DTYPE_GO> op(&pipe, &tilingData);
        op.Init(lse, go, out, outLseMax, workSpace);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UPDATE_TYPE_ONE)) {
        GET_TILING_DATA_WITH_STRUCT(AttentionUpdateTilingData, tilingData, tiling);
        AttentionUpdateWithLse<DTYPE_GO> op(&pipe, &tilingData);
        op.Init(lse, go, out, outLseMax, workSpace);
        op.Process();
    }
}
