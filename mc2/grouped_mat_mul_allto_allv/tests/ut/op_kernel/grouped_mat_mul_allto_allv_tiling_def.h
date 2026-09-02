/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MAT_MUL_ALL_TO_ALLV_TILING_DEF_H
#define GROUPED_MAT_MUL_ALL_TO_ALLV_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/grouped_mat_mul_allto_allv_tiling.h"

inline void InitAlltoAllvGroupedMatMulTilingData(uint8_t *tiling, GroupedMatMulAlltoAllvTilingData *constData)
{
    memcpy(constData, tiling, sizeof(GroupedMatMulAlltoAllvTilingData));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    GroupedMatMulAlltoAllvTilingData tilingData; \
    InitAlltoAllvGroupedMatMulTilingData(tilingArg, &tilingData)
#endif // GROUPED_MAT_MUL_ALL_TO_ALLV_TILING_DEF_H
