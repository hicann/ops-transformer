/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_sinkhorn_premix.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_sinkhorn_premix_m_k_split_core.h"
#include "mhc_pre_sinkhorn_premix_m_split_core.h"
using namespace MhcPreSinkhornPremix;

using namespace AscendC;

extern "C" __global__ __aicore__ void mhc_pre_sinkhorn_premix(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                                              GM_ADDR premix, GM_ADDR hin, GM_ADDR hPost, GM_ADDR hRes,
                                                              GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                                              GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornPremixTilingData, tiling_data_in, tiling);
    const MhcPreSinkhornPremixTilingData* __restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(0)) {
        // -----------split m k (MPMD: pre/post 与 sinkhorn 分核)------------
        TPipe pipe;
        MhcPreSinkhornPremix::MhcPreSinkhornPremixMembaseKSplitCorePart1<DTYPE_X> op;
        op.Init(x, phi, userWs, tilingData, &pipe);
        op.Process();
        pipe.Destroy();

        TPipe pipeStage2;
        MhcPreSinkhornPremix::MhcPreSinkhornPremixMembaseKSplitCorePart2<DTYPE_X, false> op2;
        op2.Init(x, alpha, bias, hin, hPost, hRes, premix, hPre, hcBeforeNorm, invRms, sumOut, normOut, userWs,
                 tilingData, &pipeStage2);
        op2.Process();
        pipeStage2.Destroy();
    } else if (TILING_KEY_IS(1)) {
        // -----------split m k (original: 单核全流程, bs<=32 或 needGrad)------------
        TPipe pipe;
        MhcPreSinkhornPremix::MhcPreSinkhornPremixMembaseKSplitCorePart1<DTYPE_X> op;
        op.Init(x, phi, userWs, tilingData, &pipe);
        op.Process();
        pipe.Destroy();

        TPipe pipeStage2;
        MhcPreSinkhornPremix::MhcPreSinkhornPremixMembaseKSplitCorePart2<DTYPE_X, true> op2;
        op2.Init(x, alpha, bias, hin, hPost, hRes, premix, hPre, hcBeforeNorm, invRms, sumOut, normOut, userWs,
                 tilingData, &pipeStage2);
        op2.Process();
        pipeStage2.Destroy();
    } else if (TILING_KEY_IS(2)) {
        // -----------split m (K=hcMult*d 超出 nd2nz srcDValue 上限, B矩阵走 MatmulImpl)------------
        TPipe pipe;
        MhcPreSinkhornPremix::MhcPreSinkhornPremixStage1<DTYPE_X> op1;
        op1.cubeCompute_.mm1_.Init(&tilingData->mm1TilingData, &pipe);
        op1.cubeCompute_.mm1_.SetSubBlockIdx(0);
        op1.Init(x, phi, invRms, hcBeforeNorm, userWs, tilingData, &pipe);
        op1.Process();
        pipe.Destroy();

        TPipe pipeStage2;
        MhcPreSinkhornPremix::MhcPreSinkhornPremixStage2<DTYPE_X> op2;
        op2.Init(x, alpha, bias, hin, hPost, hRes, premix, hPre, hcBeforeNorm, invRms, sumOut, normOut, userWs,
                 tilingData, &pipeStage2);
        op2.Process(false);
        pipeStage2.Destroy();
    }
}
