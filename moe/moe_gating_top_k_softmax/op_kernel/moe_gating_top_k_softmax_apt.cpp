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
 * \file moe_gating_top_k_softmax.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/moe_gating_top_k_softmax_tiling_def.h"
#include "arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h"

using namespace AscendC;
using namespace MoeGatingTopKSoftmax;

#define TILINGKEY_WITHOUT_FINISHED_NOT_NEED_PAD_ENGINF 100000
#define TILINGKEY_WITH_FINISHED_NOT_NEED_PAD_ENGINF 100001
#define TILINGKEY_WITHOUT_FINISHED_NEED_PAD_ENGINF 100010
#define TILINGKEY_WITH_FINISHED_NEED_PAD_ENGINF 100011

extern "C" __global__ __aicore__ void moe_gating_top_k_softmax(GM_ADDR x, GM_ADDR finished, GM_ADDR y,
                                                               GM_ADDR expertIdx, GM_ADDR rowIdx, GM_ADDR workspace,
                                                               GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    REGISTER_TILING_DEFAULT(MoeGatingTopKSoftmaxRegbaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKSoftmaxRegbaseTilingData, tilingData, tiling);

    TPipe tPipe;
    if (TILING_KEY_IS(TILINGKEY_WITHOUT_FINISHED_NOT_NEED_PAD_ENGINF)) {
        MoeGatingTopKSoftmaxFullloadGenerlized<DTYPE_X, false, false> op;
        op.Init(x, finished, y, expertIdx, rowIdx, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WITH_FINISHED_NOT_NEED_PAD_ENGINF)) {
        MoeGatingTopKSoftmaxFullloadGenerlized<DTYPE_X, true, false> op;
        op.Init(x, finished, y, expertIdx, rowIdx, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WITHOUT_FINISHED_NEED_PAD_ENGINF)) {
        MoeGatingTopKSoftmaxFullloadGenerlized<DTYPE_X, false, true> op;
        op.Init(x, finished, y, expertIdx, rowIdx, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WITH_FINISHED_NEED_PAD_ENGINF)) {
        MoeGatingTopKSoftmaxFullloadGenerlized<DTYPE_X, true, true> op;
        op.Init(x, finished, y, expertIdx, rowIdx, &tilingData, &tPipe);
        op.Process();
    }
    return;
}
