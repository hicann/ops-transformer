/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_pa_kv_cache_apt.cpp
 * \brief gather_pa_kv_cache
 */
#include "arch35/gather_pa_kv_cache_nd_arch35.h"
#include "arch35/gather_pa_kv_cache_nz.h"

#define TILING_KEY_1111 1111
#define TILING_KEY_1110 1110
#define TILING_KEY_1101 1101
#define TILING_KEY_1100 1100
#define TILING_KEY_1011 1011
#define TILING_KEY_1010 1010
#define TILING_KEY_1001 1001
#define TILING_KEY_1000 1000

extern "C" __global__ __aicore__ void gather_pa_kv_cache(GM_ADDR key_cache, GM_ADDR value_cache,
                                                         GM_ADDR block_tables, GM_ADDR seq_lens,
                                                         GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset,
                                                         GM_ADDR key_out, GM_ADDR value_out,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
// 宏封装，简化分支代码
// B8几种类型编译DataCopyPad报错，用uint8_t代替规避
#define TILING_KEY_BRANCH(tilingKey, isCacheModeNorm, isSeqLenCumSum, hasSeqOffset) {                                                           \
    if (TILING_KEY_IS(tilingKey)) {                                                                                                             \
        if constexpr (isCacheModeNorm == true) {                                                                                                \
            GatherPaKvCacheV35::GatherPaKvCacheNd<uint8_t, DTYPE_SEQ_LENS, isSeqLenCumSum, hasSeqOffset> op(&pipe, &tilingData);                \
            op.Init(key_cache, value_cache, block_tables, seq_lens, key_in, value_in, seq_offset, key_out, value_out);                          \
            op.Process();                                                                                                                       \
        } else {                                                                                                                                \
            GatherPaKvCacheV35::GatherPaKvCacheNz<uint8_t, DTYPE_SEQ_LENS, isSeqLenCumSum, hasSeqOffset> op(&pipe, &tilingData);                \
            op.Init(key_cache, value_cache, block_tables, seq_lens, key_in, value_in, seq_offset, key_out, value_out);                          \
            op.Process();                                                                                                                       \
        }                                                                                                                                       \
    }                                                                                                                                           \
} while(0);

    AscendC::TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(GatherPaKvCacheTilingDataV35, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // tilingKey, isCacheModeNorm, isSeqLenCumSum, hasSeqOffset
    TILING_KEY_BRANCH(TILING_KEY_1111, true,   true,   true)
    TILING_KEY_BRANCH(TILING_KEY_1110, true,   true,   false)
    TILING_KEY_BRANCH(TILING_KEY_1101, true,   false,  true)
    TILING_KEY_BRANCH(TILING_KEY_1100, true,   false,  false)
    TILING_KEY_BRANCH(TILING_KEY_1011, false,  true,   true)
    TILING_KEY_BRANCH(TILING_KEY_1010, false,  true,   false)
    TILING_KEY_BRANCH(TILING_KEY_1001, false,  false,  true)
    TILING_KEY_BRANCH(TILING_KEY_1000, false,  false,  false)
}