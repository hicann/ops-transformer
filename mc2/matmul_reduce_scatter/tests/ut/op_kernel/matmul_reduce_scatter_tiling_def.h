/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MATMUL_REDUCE_SCATTER_TILING_DEF_H
#define MATMUL_REDUCE_SCATTER_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "../../../../common/inc/hccl_stub.h"
#include "../../../op_kernel/matmul_reduce_scatter_tiling.h"

#define __aicore__

constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;

inline void InitMatmulReduceScatterTilingData(uint8_t* tiling, MatmulReduceScatterTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MatmulReduceScatterTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                        \
    MatmulReduceScatterTilingData tiling_data;                                                          \
    InitMatmulReduceScatterTilingData(tiling_arg, &tiling_data)
#endif  // FOREACH_MINIMUM_SCALAR_TILING_DEF_H
