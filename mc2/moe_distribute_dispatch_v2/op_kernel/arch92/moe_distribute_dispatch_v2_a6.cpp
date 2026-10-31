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
 * \file moe_distribute_dispatch_v2_a6.cpp
 * \brief A6 (arch92) kernel entry for MoeDistributeDispatchV2.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "moe_distribute_dispatch_v2_a6.h"
#include "moe_distribute_dispatch_v2_a6_tiling_key.h"
using namespace MoeDistributeDispatchV2A6Impl;
using namespace Mc2Kernel;
using namespace Mc2Tiling;
using namespace AscendC;

namespace {
template <DISPATCH_V2_A6_TEMPLATE_PARAMS>
__aicore__ inline void ExecMoeDistributeDispatchV2A6(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                                     GM_ADDR expertScales, GM_ADDR elasticInfo, GM_ADDR performanceInfo,
                                                     GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                                     GM_ADDR assistInfoOut, GM_ADDR expandScalesOut,
                                                     GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                                                     GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe* pipePtr)
{
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();

    MoeDistributeDispatchV2A6<DISPATCH_V2_A6_TEMPLATE_ARGS> op;
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
    GM_ADDR commContext = (GM_ADDR)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    op.Init(commContext, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut,
            dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, pipePtr,
            &tilingData);
    op.Process();

    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
} // namespace

template <uint8_t QuantMode, bool hasScales, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch_v2(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                                      GM_ADDR expertScales, GM_ADDR elasticInfo,
                                                      GM_ADDR performanceInfo, GM_ADDR expandXOut,
                                                      GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut,
                                                      GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                                                      GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut,
                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    TPipe pipe;
#if (ORIG_DTYPE_X == ORIG_DTYPE_EXPAND_X)
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (!hasScales && QuantMode == TILINGKEY_NO_QUANT) {
        ExecMoeDistributeDispatchV2A6<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X, Mc2Kernel::UNQUANT, false>(
            x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingGM, &pipe);
    }
#else
    if constexpr (hasScales && QuantMode == TILINGKEY_NO_QUANT) {
        ExecMoeDistributeDispatchV2A6<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X, Mc2Kernel::UNQUANT, true>(
            x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingGM, &pipe);
    }
#endif
#elif (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8)
    constexpr bool IS_DIRECT_CAST = (QuantMode == TILINGKEY_NO_QUANT) && !hasScales;
    if constexpr (IS_DIRECT_CAST || (QuantMode == TILINGKEY_STATIC_QUANT)) {
        ExecMoeDistributeDispatchV2A6<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X, QuantMode, hasScales>(
            x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingGM, &pipe);
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if constexpr (QuantMode == TILINGKEY_STATIC_QUANT || QuantMode == TILINGKEY_PERTOKEN_QUANT) {
        ExecMoeDistributeDispatchV2A6<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X, QuantMode, hasScales>(
            x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingGM, &pipe);
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT4)
    if constexpr (QuantMode == TILINGKEY_HIF4_QUANT) {
        ExecMoeDistributeDispatchV2A6<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X, QuantMode, hasScales>(
            x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingGM, &pipe);
    }
#elif ((ORIG_DTYPE_EXPAND_X == DT_FLOAT4_E2M1) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT4_E1M2))
    if constexpr (QuantMode == TILINGKEY_MX_QUANT) {
        ExecMoeDistributeDispatchV2A6<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X, QuantMode, hasScales>(
            x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingGM, &pipe);
    }
#elif (ORIG_DTYPE_DYNAMIC_SCALES == DT_FLOAT8_E8M0)
    if constexpr (QuantMode == TILINGKEY_MX_QUANT || QuantMode == TILINGKEY_MX_QUANT_CLIP) {
        ExecMoeDistributeDispatchV2A6<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X, QuantMode, hasScales>(
            x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingGM, &pipe);
    }
#elif (ORIG_DTYPE_DYNAMIC_SCALES == DT_FLOAT)
    if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT || QuantMode == TILINGKEY_PERGROUP_QUANT) {
        ExecMoeDistributeDispatchV2A6<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X, QuantMode, hasScales>(
            x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingGM, &pipe);
    }
#endif
}
