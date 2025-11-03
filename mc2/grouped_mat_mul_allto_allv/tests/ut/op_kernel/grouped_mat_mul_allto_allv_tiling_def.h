/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

inline void InitAlltoAllvGroupedMatMulTilingData(uint8_t* tiling, GroupedMatMulAlltoAllvTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(GroupedMatMulAlltoAllvTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                             \
        GroupedMatMulAlltoAllvTilingData tiling_data;                        \
        InitAlltoAllvGroupedMatMulTilingData(tiling_arg, &tiling_data)
#endif  // GROUPED_MAT_MUL_ALL_TO_ALLV_TILING_DEF_H