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
#include <cstring>

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

struct DecodeUpdateStridedTilingData {
    uint32_t formerLength = 0;
    uint32_t tailLength = 0;
    uint32_t formerNum = 0;
    uint32_t tailNum = 0;
    uint32_t hDim = 0;
    uint32_t updateType = 0;
    uint32_t sp = 0;
    uint32_t totalLength = 0;
    uint64_t rowStride = 0;
};
static_assert(sizeof(DecodeUpdateTilingData) == 32, "Legacy tiling ABI");
static_assert(sizeof(DecodeUpdateStridedTilingData) == 40, "Strided tiling ABI");

template <typename T>
inline void InitAttentionUpdateTilingData(uint8_t* tiling, T* constData)
{
    std::memcpy(constData, tiling, sizeof(T));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    DecodeUpdateTilingData tiling_data; \
    InitAttentionUpdateTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_WITH_STRUCT(type, tiling_data, tiling_arg) \
    type tiling_data; \
    InitAttentionUpdateTilingData(tiling_arg, &tiling_data)

#endif // TEST_ATTENTION_UPDATE_TILING_H
