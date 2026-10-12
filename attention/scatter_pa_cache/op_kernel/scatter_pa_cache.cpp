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
 * \file scatter_pa_cache.cpp
 * \brief scatter_pa_cache
 */

#if __has_include("../scatter_pa_kv_cache/arch35/scatter_pa_kv_cache_normal_fully_load.h")
#include "../scatter_pa_kv_cache/arch35/scatter_pa_kv_cache_normal_fully_load.h"
#include "../scatter_pa_kv_cache/arch35/scatter_pa_kv_cache_normal_not_fully_load.h"
#include "../scatter_pa_kv_cache/arch35/scatter_pa_kv_cache_rope_fully_load.h"
#include "../scatter_pa_kv_cache/arch35/scatter_pa_kv_cache_rope_not_fully_load.h"
#include "../scatter_pa_kv_cache/arch35/scatter_pa_kv_cache_alibi_fully_load.h"
#include "../scatter_pa_kv_cache/arch35/scatter_pa_kv_cache_alibi_not_fully_load.h"
#else
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_normal_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_normal_not_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_rope_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_rope_not_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_alibi_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_alibi_not_fully_load.h"
#endif

#define NORMAL_INT32_FULLY_LOAD 141
#define NORMAL_INT32_NOT_FULLY_LOAD 140

#define NORMAL_INT64_FULLY_LOAD 181
#define NORMAL_INT64_NOT_FULLY_LOAD 180

#define ROPE_INT32_FULLY_LOAD 241
#define ROPE_INT32_NOT_FULLY_LOAD 240

#define ROPE_INT64_FULLY_LOAD 281
#define ROPE_INT64_NOT_FULLY_LOAD 280

#define ALIBI_INT32_FULLY_LOAD 341
#define ALIBI_INT32_NOT_FULLY_LOAD 340

#define ALIBI_INT64_FULLY_LOAD 381
#define ALIBI_INT64_NOT_FULLY_LOAD 380

constexpr int64_t SINGLE_IN_OUT = 1;

extern "C" __global__ __aicore__ void scatter_pa_cache(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping,
                                                       GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                                       GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    if TILING_KEY_IS (NORMAL_INT32_FULLY_LOAD) {
        if constexpr (std::is_same<DTYPE_KEY, fp4x2_e2m1_t>::value || std::is_same<DTYPE_KEY, fp4x2_e1m2_t>::value ||
                      sizeof(DTYPE_KEY) == sizeof(int8_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<int8_t, int32_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<int16_t, int32_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<int32_t, int32_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        }
    } else if TILING_KEY_IS (NORMAL_INT32_NOT_FULLY_LOAD) {
        if constexpr (std::is_same<DTYPE_KEY, fp4x2_e2m1_t>::value || std::is_same<DTYPE_KEY, fp4x2_e1m2_t>::value ||
                      sizeof(DTYPE_KEY) == sizeof(int8_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<int8_t, int32_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<int16_t, int32_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<int32_t, int32_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        }
    } else if TILING_KEY_IS (NORMAL_INT64_FULLY_LOAD) {
        if constexpr (std::is_same<DTYPE_KEY, fp4x2_e2m1_t>::value || std::is_same<DTYPE_KEY, fp4x2_e1m2_t>::value ||
                      sizeof(DTYPE_KEY) == sizeof(int8_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<int8_t, int64_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<int16_t, int64_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<int32_t, int64_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        }
    } else if TILING_KEY_IS (NORMAL_INT64_NOT_FULLY_LOAD) {
        if constexpr (std::is_same<DTYPE_KEY, fp4x2_e2m1_t>::value || std::is_same<DTYPE_KEY, fp4x2_e1m2_t>::value ||
                      sizeof(DTYPE_KEY) == sizeof(int8_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<int8_t, int64_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<int16_t, int64_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
            ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<int32_t, int64_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        }
    } else if constexpr (!(std::is_same<DTYPE_KEY, fp4x2_e2m1_t>::value ||
                           std::is_same<DTYPE_KEY, fp4x2_e1m2_t>::value)) {
        if TILING_KEY_IS (ROPE_INT32_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeFullyLoad<DTYPE_KEY, int32_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if TILING_KEY_IS (ROPE_INT32_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeNotFullyLoad<DTYPE_KEY, int32_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if TILING_KEY_IS (ROPE_INT64_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeFullyLoad<DTYPE_KEY, int64_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if TILING_KEY_IS (ROPE_INT64_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeNotFullyLoad<DTYPE_KEY, int64_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if TILING_KEY_IS (ALIBI_INT32_FULLY_LOAD) {
            if constexpr (sizeof(DTYPE_KEY) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int8_t, int32_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int16_t, int32_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int32_t, int32_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            }
        } else if TILING_KEY_IS (ALIBI_INT32_NOT_FULLY_LOAD) {
            if constexpr (sizeof(DTYPE_KEY) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int8_t, int32_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                       &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int16_t, int32_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                        &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int32_t, int32_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                        &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            }
        } else if TILING_KEY_IS (ALIBI_INT64_FULLY_LOAD) {
            if constexpr (sizeof(DTYPE_KEY) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int8_t, int64_t, SINGLE_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int16_t, int64_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int32_t, int64_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            }
        } else if TILING_KEY_IS (ALIBI_INT64_NOT_FULLY_LOAD) {
            if constexpr (sizeof(DTYPE_KEY) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int8_t, int64_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                       &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int16_t, int64_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                        &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int32_t, int64_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                        &tilingData);
                op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                        key_cache_out, nullptr);
                op.Process();
            }
        }
    }
}
