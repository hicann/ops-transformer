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
 * \file scatter_pa_kv_cache.cpp
 * \brief scatter_pa_kv_cache
 */

#include "arch35/scatter_pa_kv_cache_normal_fully_load.h"
#include "arch35/scatter_pa_kv_cache_normal_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_rope_fully_load.h"
#include "arch35/scatter_pa_kv_cache_rope_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_alibi_fully_load.h"
#include "arch35/scatter_pa_kv_cache_alibi_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_omni_fully_load.h"
#include "arch35/scatter_pa_kv_cache_omni_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_nz_fully_load.h"
#include "arch35/scatter_pa_kv_cache_nz_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_norm_non_contiguous.h"
#include "arch35/scatter_pa_kv_cache_nz_non_contiguous.h"
#include "arch35/scatter_pa_kv_cache_arch35_common.h"

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

#define OMNI_FULLY_LOAD 401
#define OMNI_NOT_FULLY_LOAD 400

#define NZ_FULLY_LOAD 501
#define NZ_NOT_FULLY_LOAD 500

#define NCT_FULLY_LOAD 601
#define NCT_NOT_FULLY_LOAD 600

#define NORMAL_NON_CONTIGUOUS 700
#define NZ_NON_CONTIGUOUS 800
#define NZ_NON_CONTIGUOUS_FIVE_DIM 900

using namespace ScatterPaKvCache;

template <typename T>
struct GetInputType {
    using type = typename std::conditional<
        std::is_same<T, fp4x2_e1m2_t>::value, int8_t,
        typename std::conditional<std::is_same<T, fp4x2_e2m1_t>::value, int8_t,
            typename std::conditional<sizeof(T) == B8, int8_t,
                typename std::conditional<sizeof(T) == B16, int16_t,
                    typename std::conditional<sizeof(T) == B32, int32_t, T>::type>::type>::type>::type>::type;
};

extern "C" __global__ __aicore__ void scatter_pa_kv_cache(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping,
                                                          GM_ADDR value, GM_ADDR value_cache_in, GM_ADDR compress_lens,
                                                          GM_ADDR compress_seq_offset, GM_ADDR seq_lens,
                                                          GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    using keyType = GetInputType<DTYPE_KEY>::type;
    using valueType = GetInputType<DTYPE_VALUE>::type;
    if TILING_KEY_IS (NORMAL_INT32_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<keyType, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS (NORMAL_INT32_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<keyType, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS (NORMAL_INT64_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<keyType, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS (NORMAL_INT64_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<keyType, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS(NORMAL_NON_CONTIGUOUS) {
        ScatterPaKvCache::ScatterPaKvCacheNormNonContiguous<keyType, DTYPE_SLOT_MAPPING, DUAL_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS(NZ_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNzFullyLoad<keyType, valueType, DTYPE_SLOT_MAPPING, DUAL_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS(NZ_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNzNotFullyLoad<keyType, valueType, DTYPE_SLOT_MAPPING, DUAL_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS(NZ_NON_CONTIGUOUS) {
        ScatterPaKvCache::ScatterPaKvCacheNzNonContiguous<keyType, valueType, DTYPE_SLOT_MAPPING, DUAL_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    }  else if TILING_KEY_IS(NZ_NON_CONTIGUOUS_FIVE_DIM) {
        ScatterPaKvCache::ScatterPaKvCacheNzNonContiguous<keyType, valueType, DTYPE_SLOT_MAPPING, DUAL_IN_OUT, true> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS (NCT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<keyType, DTYPE_SLOT_MAPPING, DUAL_IN_OUT, true> op(
            &pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if TILING_KEY_IS (NCT_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<keyType, DTYPE_SLOT_MAPPING, DUAL_IN_OUT, true> op(
            &pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if constexpr (!(std::is_same<DTYPE_KEY, fp4x2_e2m1_t>::value ||
                           std::is_same<DTYPE_KEY, fp4x2_e1m2_t>::value)) {
        if TILING_KEY_IS (ROPE_INT32_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeFullyLoad<DTYPE_KEY, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if TILING_KEY_IS (ROPE_INT32_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeNotFullyLoad<DTYPE_KEY, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if TILING_KEY_IS (ROPE_INT64_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeFullyLoad<DTYPE_KEY, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if TILING_KEY_IS (ROPE_INT64_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeNotFullyLoad<DTYPE_KEY, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if TILING_KEY_IS (OMNI_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheOmniFullyLoad<DTYPE_KEY, DTYPE_SLOT_MAPPING, DUAL_IN_OUT> op(&pipe,
                                                                                                           &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if TILING_KEY_IS (OMNI_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheOmniNotFullyLoad<DTYPE_KEY, DTYPE_SLOT_MAPPING, DUAL_IN_OUT> op(
                &pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if TILING_KEY_IS (ALIBI_INT32_FULLY_LOAD) {
            if constexpr (sizeof(DTYPE_KEY) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int8_t, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int16_t, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int32_t, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            }
        } else if TILING_KEY_IS (ALIBI_INT32_NOT_FULLY_LOAD) {
            if constexpr (sizeof(DTYPE_KEY) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int8_t, int32_t, DUAL_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int16_t, int32_t, DUAL_IN_OUT> op(&pipe,
                                                                                                      &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int32_t, int32_t, DUAL_IN_OUT> op(&pipe,
                                                                                                      &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            }
        } else if TILING_KEY_IS (ALIBI_INT64_FULLY_LOAD) {
            if constexpr (sizeof(DTYPE_KEY) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int8_t, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int16_t, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int32_t, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            }
        } else if TILING_KEY_IS (ALIBI_INT64_NOT_FULLY_LOAD) {
            if constexpr (sizeof(DTYPE_KEY) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int8_t, int64_t, DUAL_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int16_t, int64_t, DUAL_IN_OUT> op(&pipe,
                                                                                                      &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(DTYPE_KEY) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int32_t, int64_t, DUAL_IN_OUT> op(&pipe,
                                                                                                      &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            }
        }
    }
}