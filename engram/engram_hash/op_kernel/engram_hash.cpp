/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_hash.cpp
 * \brief EngramHash kernel 入口
 */

#include "engram_hash_kernel.h"

using namespace AscendC;

// dtype 宏缺省兜底: 正式编译由 binary.json 按 dtype 组合下发, UT 由 CMake 显式传入。
#ifndef DTYPE_NGRAM_TOKEN_IDS
#define DTYPE_NGRAM_TOKEN_IDS int32_t
#endif
#ifndef DTYPE_MULTIPLIERS
#define DTYPE_MULTIPLIERS int64_t
#endif
#ifndef DTYPE_VOCAB_SIZES
#define DTYPE_VOCAB_SIZES int32_t
#endif
#ifndef DTYPE_OFFSETS
#define DTYPE_OFFSETS int32_t
#endif
#ifndef DTYPE_HASH_INDICES
#define DTYPE_HASH_INDICES int32_t
#endif

extern "C" __global__ __aicore__ void engram_hash(GM_ADDR ngram_token_ids, GM_ADDR multipliers, GM_ADDR vocab_sizes,
                                                  GM_ADDR offsets, GM_ADDR hash_indices, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GET_TILING_DATA_WITH_STRUCT(EngramHashTilingData, tilingDataIn, tiling);

    if (TILING_KEY_IS(0)) {
        // 空输入 (numTokens == 0): 直接返回, 输出保持为空 tensor。
        if (tilingDataIn.numTokens == 0) {
            return;
        }
        asc_init();
        EngramHashOps::EngramHashProcess<DTYPE_NGRAM_TOKEN_IDS, DTYPE_MULTIPLIERS, DTYPE_HASH_INDICES>(
            ngram_token_ids, multipliers, vocab_sizes, offsets, hash_indices, &tilingDataIn);
    }
}
