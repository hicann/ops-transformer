/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_DEF_H
#define ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "../../../../common/inc/hccl_stub.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_tiling.h"

#define __aicore__

inline void InitAlltoAllvGroupedMatMulTilingData(uint8_t* tiling, AlltoAllvGmmTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(AlltoAllvGmmTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    AlltoAllvGmmTilingData tiling_data;          \
    InitAlltoAllvGroupedMatMulTilingData(tiling_arg, &tiling_data)
#endif  // ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_DEF_H
