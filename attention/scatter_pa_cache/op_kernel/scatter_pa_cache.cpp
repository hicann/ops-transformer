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
#include "../scatter_pa_kv_cache/arch35/scatter_pa_kv_cache_arch35_common.h"
#else
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_normal_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_normal_not_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_rope_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_rope_not_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_alibi_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_alibi_not_fully_load.h"
#include "../../scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_arch35_common.h"
#endif

#define NORMAL_INT32_FULLY_LOAD 141
#define NORMAL_INT64_FULLY_LOAD 181

#define NORMAL_NOT_FULLY_LOAD 100

#define ROPE_FULLY_LOAD 201
#define ROPE_NOT_FULLY_LOAD 200

#define ALIBI_FULLY_LOAD 301
#define ALIBI_NOT_FULLY_LOAD 300

constexpr int64_t SINGLE_IN_OUT = 1;

template <typename T>
struct GetInputType {
    using type = typename std::conditional<
        std::is_same<T, fp4x2_e1m2_t>::value, int8_t,
        typename std::conditional<
            std::is_same<T, fp4x2_e2m1_t>::value, int8_t,
            typename std::conditional<
                std::is_same<T, hifloat4x2_t>::value, int8_t,
                typename std::conditional<
                    sizeof(T) == ScatterPaKvCache::B8, int8_t,
                    typename std::conditional<sizeof(T) == ScatterPaKvCache::B16, int16_t,
                                              typename std::conditional<sizeof(T) == ScatterPaKvCache::B32, int32_t,
                                                                        T>::type>::type>::type>::type>::type>::type;
};

extern "C" __global__ __aicore__ void scatter_pa_cache(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping,
                                                       GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                                       GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    using keyType = GetInputType<DTYPE_KEY>::type;
    if TILING_KEY_IS (NORMAL_INT32_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<keyType, int32_t, int32_t, SINGLE_IN_OUT> op(&pipe,
                                                                                                       &tilingData);
        op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                key_cache_out, nullptr);
        op.Process();
    } else if TILING_KEY_IS (NORMAL_INT64_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<keyType, DTYPE_SLOT_MAPPING, int64_t, SINGLE_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                key_cache_out, nullptr);
        op.Process();
    } else if TILING_KEY_IS (NORMAL_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<keyType, DTYPE_SLOT_MAPPING, SINGLE_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                key_cache_out, nullptr);
        op.Process();
    } else if constexpr (!(std::is_same<DTYPE_KEY, fp4x2_e2m1_t>::value ||
                           std::is_same<DTYPE_KEY, fp4x2_e1m2_t>::value ||
                           std::is_same<DTYPE_KEY, hifloat4x2_t>::value)) {
        if TILING_KEY_IS (ROPE_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeFullyLoad<DTYPE_KEY, DTYPE_SLOT_MAPPING, SINGLE_IN_OUT> op(
                &pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if TILING_KEY_IS (ROPE_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeNotFullyLoad<DTYPE_KEY, DTYPE_SLOT_MAPPING, SINGLE_IN_OUT> op(
                &pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if TILING_KEY_IS (ALIBI_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<keyType, DTYPE_SLOT_MAPPING, SINGLE_IN_OUT> op(
                &pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        } else if TILING_KEY_IS (ALIBI_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<keyType, DTYPE_SLOT_MAPPING, SINGLE_IN_OUT> op(
                &pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, nullptr, nullptr, compress_lens, compress_seq_offset, seq_lens,
                    key_cache_out, nullptr);
            op.Process();
        }
    }
}
