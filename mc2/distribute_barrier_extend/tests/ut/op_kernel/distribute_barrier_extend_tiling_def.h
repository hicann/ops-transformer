/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISTRIBUTE_BARRIER_TILING_DEF_H
#define DISTRIBUTE_BARRIER_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"
#include "../../../../../tests/ut/framework_normal/common/hccl_stub.h"
#include "../../../op_kernel/distribute_barrier_tiling.h"

inline void InitDistributeBarrierTilingData(uint8_t *tiling, DistributeBarrierTilingData *constData)
{
    memcpy_s(constData, sizeof(DistributeBarrierTilingData), tiling, sizeof(DistributeBarrierTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(DistributeBarrierTilingData, tilingData, tilingArg) \
    do { \
        DistributeBarrierTilingData tilingData; \
        InitDistributeBarrierTilingData(tilingArg, &(tilingData)) \
    } while (0)

#endif
