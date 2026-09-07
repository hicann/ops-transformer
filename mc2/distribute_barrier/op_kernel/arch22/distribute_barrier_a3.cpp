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
 * \file distribute_barrier_a3.cpp
 * \brief A3 (arch22) kernel entry for DistributeBarrier.
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../distribute_barrier_tiling.h"
#include "../distribute_barrier.h"

using namespace AscendC;
using namespace DistributeBarrierImpl;

extern "C" __global__ __aicore__ void distribute_barrier(GM_ADDR xRef, GM_ADDR timeOut, GM_ADDR elasticInfo,
                                                         GM_ADDR xRefOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DistributeBarrierTilingData);
    GET_TILING_DATA_WITH_STRUCT(DistributeBarrierTilingData, tilingData, tilingGM);
    TPipe pipe;
#if ((ORIG_DTYPE_X_REF == DT_BF16) || (ORIG_DTYPE_X_REF == DT_FLOAT16) || (ORIG_DTYPE_X_REF == DT_FLOAT) || \
     (ORIG_DTYPE_X_REF == DT_BOOL) || (ORIG_DTYPE_X_REF == DT_INT8) || (ORIG_DTYPE_X_REF == DT_INT16) || \
     (ORIG_DTYPE_X_REF == DT_INT32) || (ORIG_DTYPE_X_REF == DT_INT64) || (ORIG_DTYPE_X_REF == DT_UINT8) || \
     (ORIG_DTYPE_X_REF == DT_UINT16) || (ORIG_DTYPE_X_REF == DT_UINT32) || (ORIG_DTYPE_X_REF == DT_UINT64))
    DistributeBarrier<DTYPE_X_REF> op;
    op.Init(nullptr, timeOut, elasticInfo, workspaceGM, &pipe, &tilingData);
    op.Process();
#endif
}
