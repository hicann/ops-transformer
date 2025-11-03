/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _ROPE_WITH_SIN_COS_CACHE_H
#define _ROPE_WITH_SIN_COS_CACHE_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define ORIG_DTYPE_QUERYIN DT_FLOAT

#define __aicore__

inline void InitRopeWithSinCosCacheTilingData(uint8_t* tiling, RopeWithSinCosCacheTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(RopeWithSinCosCacheTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    RopeWithSinCosCacheTilingData tilingData;      \
    InitRopeWithSinCosCacheTilingData(tilingPointer, &tilingData)
#endif
