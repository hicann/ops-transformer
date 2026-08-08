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
 * \file engram_fetch_grad.cpp
 * \brief EngramFetchGrad 算子 kernel 入口
 */

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_ENGRAM_FETCH_GRAD_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "engram_fetch_grad_tiling_data.h"
#include "engram_fetch_grad_tiling_key.h"
#include "arch35/engram_fetch_grad_arch35.h"
#include "engram_fetch_grad_utils.h"

using namespace Mc2Kernel;

template <uint32_t EngramFetchGradMode>
__global__ __aicore__ void engram_fetch_grad(GM_ADDR commContext, GM_ADDR gradFetched, GM_ADDR perm, GM_ADDR sendCounts,
                                             GM_ADDR recvCounts, GM_ADDR recvLocalEntry, GM_ADDR numRecv,
                                             GM_ADDR gradUniqueOut, GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut,
                                             GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(EngramFetchGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(EngramFetchGradTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

#if defined(ENABLE_ENGRAM_FETCH_GRAD_KERNEL)
    TPipe pipe;
    EngramFetchGradArch35 op;
    op.Init(commContext, gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv,
            gradUniqueOut, uniqueLocalEntryOut, numUniqueOut, workspaceGM, &pipe, &tilingData);
    op.Process();
#endif
}
