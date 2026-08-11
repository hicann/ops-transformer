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
 * \file und_gen_qkv_rms_norm_rope_cache.cpp
 * \brief kernel 入口（仅 Ascend 950 / arch35）
 */

#include "arch35/und_gen_qkv_rms_norm_rope_cache_regbase.h"

using namespace UndGenQkvRmsNormRopeCache;

extern "C" __global__ __aicore__ void und_gen_qkv_rms_norm_rope_cache(
    // 必选输入（8 个）
    GM_ADDR und_qkv, GM_ADDR und_weights_q, GM_ADDR und_weights_k, GM_ADDR cos_sin_cache, GM_ADDR k_cache,
    GM_ADDR v_cache, GM_ADDR slot_mapping, GM_ADDR positions,
    // 可选输入（4 个）
    GM_ADDR gen_qkv, GM_ADDR gen_weights_q, GM_ADDR gen_weights_k, GM_ADDR cat_indices,
    // 输出（3 个，k_cache/v_cache 与入参同地址，原地写入）
    GM_ADDR q, GM_ADDR k_cache_out, GM_ADDR v_cache_out,
    // workspace + tiling
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA_WITH_STRUCT(UndGenQkvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const UndGenQkvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        UndGenQkvRmsNormRopeCacheRegbase<DTYPE_UND_QKV, DTYPE_K_CACHE> op(&pipe, tilingData);
        op.Init(und_qkv, und_weights_q, und_weights_k, cos_sin_cache, k_cache, v_cache, slot_mapping, positions,
                gen_qkv, gen_weights_q, gen_weights_k, cat_indices, q);
        op.Process();
    }
}
