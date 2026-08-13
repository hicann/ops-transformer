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
 * \file moe_distribute_dispatch_v2_apt.cpp
 * \brief A5 (arch35) kernel entry for MoeDistributeDispatchV2.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../moe_distribute_dispatch_v2.h"
#include "moe_distribute_dispatch_v2_a5_full_mesh.h"
#include "moe_distribute_dispatch_v2_apt_tiling_key.h"
#include "moe_distribute_dispatch_v2_a5_ccu.h"
using namespace MoeDistributeDispatchA5Impl;
using namespace MoeDistributeDispatchV2A5FullMeshImpl;
using namespace Mc2Kernel;
using namespace Mc2Tiling;
using namespace AscendC;

template <uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void
moe_distribute_dispatch_v2(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
                           GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                           GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                           GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
    TPipe pipe;
    GM_ADDR contextGM0 = (GM_ADDR)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (CommMode == TILINGKEY_TPL_CCU) {
            MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, Mc2Kernel::UNQUANT, false> op;
            op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        } else if constexpr (CommMode == TILINGKEY_TPL_MTE) {
            if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
                MoeDistributeDispatchV2A5FullMesh<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X,
                                                  Mc2Kernel::UNQUANT, false>
                    op;
                op.Init(contextGM0, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                        expandXOut, dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut,
                        epSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            } else {
                MoeDistributeDispatchV2<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X,
                                        Mc2Kernel::UNQUANT, false>
                    op;
                op.Init(contextGM0, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                        expandXOut, dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut,
                        epSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            }
        }
    }
#elif ((ORIG_DTYPE_X == DT_FLOAT8_E5M2) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E5M2)) || \
    ((ORIG_DTYPE_X == DT_FLOAT8_E4M3FN) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E4M3FN)) || \
    ((ORIG_DTYPE_X == DT_HIFLOAT8) && (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8)) || \
    ((ORIG_DTYPE_X == DT_FLOAT4_E2M1) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT4_E2M1)) || \
    ((ORIG_DTYPE_X == DT_FLOAT4_E1M2) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT4_E1M2))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (CommMode == TILINGKEY_TPL_MTE) {
            if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
                MoeDistributeDispatchV2A5FullMesh<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X,
                                                  Mc2Kernel::UNQUANT, true>
                    op;
                op.Init(contextGM0, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                        expandXOut, dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut,
                        epSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            } else {
                MoeDistributeDispatchV2<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X,
                                        Mc2Kernel::UNQUANT, true>
                    op;
                op.Init(contextGM0, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                        expandXOut, dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut,
                        epSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            }
        }
    }
#elif ((ORIG_DTYPE_EXPAND_X == DT_INT8) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E5M2) || \
       (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT4_E2M1) || \
       (ORIG_DTYPE_EXPAND_X == DT_FLOAT4_E1M2))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (QuantMode != TILINGKEY_NO_QUANT) {
            if constexpr (CommMode == TILINGKEY_TPL_CCU) {
                MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, QuantMode, ScaleMode> op;
                op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            } else if constexpr (CommMode == TILINGKEY_TPL_MTE) {
                if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
                    MoeDistributeDispatchV2A5FullMesh<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X,
                                                      QuantMode, ScaleMode>
                        op;
                    op.Init(contextGM0, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                            expandXOut, dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut,
                            epSendCountsOut, workspaceGM, &pipe, &tilingData);
                    op.Process();
                } else {
                    MoeDistributeDispatchV2<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X,
                                            QuantMode, ScaleMode>
                        op;
                    op.Init(contextGM0, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                            expandXOut, dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut,
                            epSendCountsOut, workspaceGM, &pipe, &tilingData);
                    op.Process();
                }
            }
        }
    }
#elif ((ORIG_DTYPE_X == DT_BF16) && (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8)) || \
    ((ORIG_DTYPE_X == DT_FLOAT16) && (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (CommMode == TILINGKEY_TPL_CCU) {
            MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, Mc2Kernel::STATIC_QUANT, true> op;
            op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        } else if constexpr (CommMode == TILINGKEY_TPL_MTE) {
            if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
                MoeDistributeDispatchV2A5FullMesh<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X,
                                                  QuantMode, ScaleMode>
                    op;
                op.Init(contextGM0, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                        expandXOut, dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut,
                        epSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            } else {
                MoeDistributeDispatchV2<Mc2Kernel::HcclContextHolder, DTYPE_X, DTYPE_EXPAND_X,
                                        QuantMode, ScaleMode>
                    op;
                op.Init(contextGM0, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                        expandXOut, dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut,
                        epSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            }
        }
    }
#endif
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
