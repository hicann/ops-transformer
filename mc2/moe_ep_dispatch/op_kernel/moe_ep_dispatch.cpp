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
 * \file moe_ep_dispatch.cpp
 * \brief
 */

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_tiling_key.h"
#include "moe_ep_dispatch_tiling.h"
#include "moe_ep_dispatch.h"
#include "moe_ep_dispatch_hybrid.h"

using namespace MoeEpDispatchImpl;
using namespace Mc2Tiling;
using namespace AscendC;

template <bool DoCpuSync, bool IsCached, bool IsTopkWeights, bool IsMxQuant, uint8_t NetworkMode>
__global__ __aicore__ void moe_ep_dispatch(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
                                           GM_ADDR scales, GM_ADDR cachedSlotIdx, GM_ADDR cachedRouteCount,
                                           GM_ADDR cachedRouteDstScaleout, GM_ADDR cachedRouteScaleoutSlot,
                                           GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx,
                                           GM_ADDR routeCount, GM_ADDR routeDstScaleout, GM_ADDR routeScaleoutSlot,
                                           GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeEpDispatchTilingData);

    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeEpDispatchTilingData, tilingData, tilingGM);
    using ScalesType = typename std::conditional<IsMxQuant, fp8_e8m0_t, float>::type;

#if defined(ENABLE_MOE_EP_KERNEL)

    if constexpr (NetworkMode == TILINGKEY_HYBRID) {
        MoeEpDispatchHybridImpl::MoeEpDispatchHybrid<DTYPE_X, ScalesType, DoCpuSync, IsCached, IsTopkWeights,
                                                     NetworkMode>
            op;
        op.Init(context, x, topkIdx, topkWeights, scales, cachedSlotIdx, cachedRouteCount, cachedRouteDstScaleout,
                cachedRouteScaleoutSlot, numRecvPerRank, numRecvPerExpert, dstBufferSlotIdx, routeCount,
                routeDstScaleout, routeScaleoutSlot, workspaceGM, tilingGM, &pipe, &tilingData);
        op.Process();
    } else {
        MoeEpDispatch<DTYPE_X, ScalesType, DoCpuSync, IsCached, IsTopkWeights, NetworkMode> op;
        op.Init(context, x, topkIdx, topkWeights, scales, cachedSlotIdx, numRecvPerRank, numRecvPerExpert,
                dstBufferSlotIdx, workspaceGM, tilingGM, &pipe, &tilingData);
        op.Process();
    }

#endif
}
