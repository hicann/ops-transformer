/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_MOE_FINALIZE_ROUTING_V2_GRAD_TILING_H
#define TEST_MOE_FINALIZE_ROUTING_V2_GRAD_TILING_H

#include "kernel_tiling/kernel_tiling.h"

#define DT_FLOAT float
#define DT_FLOAT16 half
#define DT_BF16 bfloat16_t
#define DT_INT32 int32_t
#define DTYPE_GRAD_Y float
#define ORIG_DTYPE_GRAD_Y DT_FLOAT
#define DTYPE_EXPANDED_ROW_IDX int32_t
#define ORIG_DTYPE_EXPANDED_ROW_IDX DT_INT32

#define __CCE_UT_TEST__
#define __CCE_AICORE__ 220

inline void InitMoeFinalizeRoutingV2GradTilingData(uint8_t *tiling, MoeFinalizeRoutingV2GradTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeFinalizeRoutingV2GradTilingData));
}

#undef GET_TILING_DATA
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    MoeFinalizeRoutingV2GradTilingData tiling_data; \
    InitMoeFinalizeRoutingV2GradTilingData(tiling_arg, &tiling_data)

#endif
