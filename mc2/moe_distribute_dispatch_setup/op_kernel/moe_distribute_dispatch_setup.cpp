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
 * \file moe_distribute_dispatch_setup.cpp
 * \brief kernel内核实现
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "moe_distribute_dispatch_setup_tiling.h"
#include "arch35/moe_distribute_dispatch_setup_arch35.h"

using namespace AscendC;
using namespace Mc2Kernel;
extern "C" __global__ __aicore__ void moe_distribute_dispatch_setup(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
                                                                    GM_ADDR xActiveMask, GM_ADDR YOut,
                                                                    GM_ADDR expandIdxOut, GM_ADDR commCmdInfoOut,
                                                                    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchSetupTilingData);
    TPipe pipe;
    auto tiling = (__gm__ MoeDistributeDispatchSetupTilingData *)tilingGM;
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#if (ORIG_DTYPE_Y == DT_BF16 || ORIG_DTYPE_Y == DT_FLOAT16)
    if (TILING_KEY_IS(1000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
        MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, UNQUANT, false> op;
        op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#elif ((ORIG_DTYPE_Y == DT_INT8) || (ORIG_DTYPE_Y == DT_FLOAT8_E5M2) || (ORIG_DTYPE_Y == DT_FLOAT8_E4M3FN) || \
       (ORIG_DTYPE_Y == DT_HIFLOAT8))
    if (TILING_KEY_IS(1011)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
        MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, STATIC_QUANT, true> op;
        op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
        MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, PERTOKEN_DYNAMIC_QUANT, false> op;
        op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
        MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, PERTOKEN_DYNAMIC_QUANT, true> op;
        op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1003)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
        MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, PERGROUP_DYNAMIC_QUANT, false> op;
        op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1013)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
        MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, PERGROUP_DYNAMIC_QUANT, true> op;
        op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1004)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
        MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, MX_QUANT, false> op;
        op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }

#endif
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
