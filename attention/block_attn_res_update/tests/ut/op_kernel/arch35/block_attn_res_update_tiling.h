/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_BLOCK_ATTN_RES_UPDATE_TILING_H
#define TEST_BLOCK_ATTN_RES_UPDATE_TILING_H

#include <cstdint>
#include <cstring>

#ifdef __CCE_KT_TEST__
#include "block_attn_res_update_cpu_debug_stub.h"
#endif

#include "../../../../op_kernel/arch35/block_attn_res_update_tiling_data.h"

inline void InitBlockAttnResUpdateTilingData(const uint8_t *tiling, BlockAttnResUpdateTilingData *tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(BlockAttnResUpdateTilingData));
}

#ifdef GET_TILING_DATA_WITH_STRUCT
#undef GET_TILING_DATA_WITH_STRUCT
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData; \
    InitBlockAttnResUpdateTilingData(reinterpret_cast<const uint8_t *>(tilingArg), &tilingData)

#endif // TEST_BLOCK_ATTN_RES_UPDATE_TILING_H
