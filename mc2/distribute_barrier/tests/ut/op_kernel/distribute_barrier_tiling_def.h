/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef DISTRIBUTE_BARRIER_TILING_DEF_H
#define DISTRIBUTE_BARRIER_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"
#include "../../../../common/inc/hccl_stub.h"
#include "../../../op_kernel/distribute_barrier_tiling.h"

inline void InitDistributeBarrierTilingData(uint8_t* tiling, DistributeBarrierTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(DistributeBarrierTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(DistributeBarrierTilingData, tiling_data, tiling_arg)       \
    DistributeBarrierTilingData tiling_data;                                                 \
    InitDistributeBarrierTilingData(tiling_arg, &tiling_data)

#endif