/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file distribute_barrier.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "distribute_barrier_tiling.h"
#include "distribute_barrier.h"

using namespace AscendC;
using namespace DistributeBarrierImpl;

extern "C" __global__ __aicore__ void distribute_barrier(GM_ADDR xRef, GM_ADDR timeOut, GM_ADDR elasticInfo,
                                                         GM_ADDR xRefOut, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
  REGISTER_TILING_DEFAULT(DistributeBarrierTilingData);
  TPipe pipe;

  if (TILING_KEY_IS(10000)) {
    GET_TILING_DATA_WITH_STRUCT(DistributeBarrierTilingData, tilingData, tilingGM);
    DistributeBarrier<DTYPE_X_REF> op;
    op.Init(timeOut, elasticInfo, workspaceGM, &pipe, &tilingData);
    op.Process();
  }
}