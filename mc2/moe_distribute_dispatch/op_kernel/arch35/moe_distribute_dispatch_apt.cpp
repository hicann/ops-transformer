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
 * \file moe_distribute_dispatch_apt.cpp
 * \brief A5 (arch35) kernel entry for MoeDistributeDispatch.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "moe_distribute_dispatch_apt_tiling_key.h"
#include "../moe_distribute_dispatch.h"
using namespace MoeDistributeDispatchImpl;
using namespace Mc2Tiling;
using namespace AscendC;

template <uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                                   GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                                   GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                                   GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut,
                                                   GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
    TPipe pipe;
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (QuantMode == TILINGKEY_NO_QUANT) {
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, ScaleMode> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}
