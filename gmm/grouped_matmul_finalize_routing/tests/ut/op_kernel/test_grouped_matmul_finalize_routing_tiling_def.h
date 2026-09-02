/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_GROUPED_MATMUL_FRTILING_TILING_H_
#define _TEST_GROUPED_MATMUL_FRTILING_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

inline void InitGroupedMatmulFRTilingData(uint8_t *tiling, GroupMatmulFRTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(GroupMatmulFRTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    GroupMatmulFRTilingData tiling_data; \
    InitGroupedMatmulFRTilingData(tiling_arg, &tiling_data)
#endif
