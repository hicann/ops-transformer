/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SwinTransformerLnQkvQuant_TILING_H__
#define __SwinTransformerLnQkvQuant_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#ifdef __NPU_TILING__
inline[aicore] void InitSwinTransformerLnQkvQuantTilingData(const __gm__ uint8_t *tiling,
                                                            SwinTransformerLnQkvQuantTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(SwinTransformerLnQkvQuantTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitSwinTransformerLnQkvQuantTilingData(uint8_t *tiling, SwinTransformerLnQkvQuantTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(SwinTransformerLnQkvQuantTilingData));
}
#endif

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    SwinTransformerLnQkvQuantTilingData tiling_data; \
    InitSwinTransformerLnQkvQuantTilingData(tiling_arg, &tiling_data)

#endif
