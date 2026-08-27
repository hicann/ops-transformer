/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre.cpp
 * \brief
 */

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_m_split_core.h"
#include "mhc_pre_base.h"
using namespace MhcPre;

using namespace AscendC;

extern "C" __global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma,
                                              GM_ADDR hin, GM_ADDR hPost, GM_ADDR hRes, GM_ADDR invRms, GM_ADDR hMix,
                                              GM_ADDR hPre, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(MhcPreMembaseTilingData, tiling_data_in, tiling);
    const MhcPreMembaseTilingData *__restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(0)) {
        // -----------power + res--------------
        TPipe pipe;
        MhcPre::MhcPreStage1<DTYPE_X, true, true> op1;
        op1.cubeCompute_.mm1_.Init(&tilingData->mm1TilingData, &pipe);
        op1.cubeCompute_.mm1_.SetSubBlockIdx(0);
        op1.Init(x, phi, gamma, invRms, hMix, userWs, tilingData, &pipe);
        op1.Process();
        pipe.Destroy();

        TPipe pipeStage2;
        MhcPre::MhcPreStage2<DTYPE_X, true, true> op2;
        op2.Init(x, alpha, bias, hin, hPost, hRes, hPre, hMix, invRms, userWs, tilingData, &pipeStage2);
        op2.Process(false);
        pipeStage2.Destroy();
    } else if (TILING_KEY_IS(10)) {
        // -----------factorial + res--------------
        TPipe pipe;
        MhcPre::MhcPreStage1<DTYPE_X, false, true> opStage1;
        opStage1.cubeCompute_.mm1_.Init(&tilingData->mm1TilingData, &pipe);
        opStage1.cubeCompute_.mm1_.SetSubBlockIdx(0);
        opStage1.Init(x, phi, gamma, invRms, hMix, userWs, tilingData, &pipe);
        opStage1.Process();
        pipe.Destroy();

        TPipe pipeStage2;
        MhcPre::MhcPreStage2<DTYPE_X, false, true> opStage2;
        opStage2.Init(x, alpha, bias, hin, hPost, hRes, hPre, hMix, invRms, userWs, tilingData, &pipeStage2);
        opStage2.Process(false);
        pipeStage2.Destroy();
    } else if (TILING_KEY_IS(11)) {
        // -----------factorial no res--------------
        TPipe pipe;
        MhcPre::MhcPreStage1<DTYPE_X, false, false> opStage1;
        opStage1.cubeCompute_.mm1_.Init(&tilingData->mm1TilingData, &pipe);
        opStage1.cubeCompute_.mm1_.SetSubBlockIdx(0);
        opStage1.Init(x, phi, gamma, invRms, hMix, userWs, tilingData, &pipe);
        opStage1.Process();
        pipe.Destroy();

        TPipe pipeStage2;
        MhcPre::MhcPreStage2<DTYPE_X, false, false> opStage2;
        opStage2.Init(x, alpha, bias, hin, hPost, hRes, hPre, hMix, invRms, userWs, tilingData, &pipeStage2);
        opStage2.Process(false);
        pipeStage2.Destroy();
    }
    // #endif
}
#endif
