/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_ATTENTION_UPDATE_TILING_H
#define TEST_ATTENTION_UPDATE_TILING_H

#include <cstddef>
#include <cstdint>

#pragma pack(1)
struct DecodeUpdateTilingData {
    uint32_t formerLength = 0;
    uint32_t tailLength = 0;
    uint32_t formerNum = 0;
    uint32_t tailNum = 0;
    uint32_t hDim = 0;
    uint32_t updateType = 0;
    uint32_t sp = 0;
    uint32_t totalLength = 0;
};
#pragma pack()

inline void InitAttentionUpdateTilingData(uint8_t *tiling, DecodeUpdateTilingData *constData)
{
    auto *src = reinterpret_cast<uint32_t *>(tiling);
    auto *dst = reinterpret_cast<uint32_t *>(constData);
    for (size_t i = 0; i < sizeof(DecodeUpdateTilingData) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    DecodeUpdateTilingData tiling_data; \
    InitAttentionUpdateTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_WITH_STRUCT(type, tiling_data, tiling_arg) \
    type tiling_data; \
    InitAttentionUpdateTilingData(tiling_arg, &tiling_data)

#endif // TEST_ATTENTION_UPDATE_TILING_H
