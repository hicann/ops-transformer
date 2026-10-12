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
 * \file indexer_quant_cache.cpp
 * \brief
 */

#include "arch35/indexer_quant_cache_multi_row.h"
#include "arch35/indexer_quant_cache_single_row.h"
#include "arch35/indexer_quant_cache_multi_row_mx_fp8.h"
#include "arch35/indexer_quant_cache_single_row_mx_fp8.h"
#include "arch35/indexer_quant_cache_multi_row_mx_fp4.h"
#include "arch35/indexer_quant_cache_single_row_mx_fp4.h"

#define SINGLE_ROW_NORMAL_QUANT 10001
#define SINGLE_ROW_MXFP8_QUANT 10000
#define SINGLE_ROW_MXFP4_QUANT 10002
#define MULTI_ROW_NORMAL_QUANT 10011
#define MULTI_ROW_MXFP8_QUANT 10010
#define MULTI_ROW_MXFP4_QUANT 10012

using namespace AscendC;

extern "C" __global__ __aicore__ void indexer_quant_cache(GM_ADDR cache, GM_ADDR cache_scale, GM_ADDR x,
                                                          GM_ADDR slot_mapping, GM_ADDR cache_out,
                                                          GM_ADDR cache_scale_out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if (TILING_KEY_IS(MULTI_ROW_NORMAL_QUANT)) {
        IndexerQuantCache::IndexerQuantCacheMultiRow<DTYPE_X, DTYPE_CACHE, DTYPE_CACHE_SCALE> op;
        op.Init(x, slot_mapping, cache, cache_scale, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(SINGLE_ROW_NORMAL_QUANT)) {
        IndexerQuantCache::IndexerQuantCacheSingleRow<DTYPE_X, DTYPE_CACHE, DTYPE_CACHE_SCALE> op;
        op.Init(x, slot_mapping, cache, cache_scale, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(MULTI_ROW_MXFP8_QUANT)) {
        IndexerQuantCache::IndexerQuantCacheMultiRowMxFp8<DTYPE_X, DTYPE_CACHE, DTYPE_CACHE_SCALE> op;
        op.Init(x, slot_mapping, cache, cache_scale, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(SINGLE_ROW_MXFP8_QUANT)) {
        IndexerQuantCache::IndexerQuantCacheSingleRowMxFp8<DTYPE_X, DTYPE_CACHE, DTYPE_CACHE_SCALE> op;
        op.Init(x, slot_mapping, cache, cache_scale, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(MULTI_ROW_MXFP4_QUANT)) {
        IndexerQuantCache::IndexerQuantCacheMultiRowMxFp4<DTYPE_X, DTYPE_CACHE, DTYPE_CACHE_SCALE> op;
        op.Init(x, slot_mapping, cache, cache_scale, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(SINGLE_ROW_MXFP4_QUANT)) {
        IndexerQuantCache::IndexerQuantCacheSingleRowMxFp4<DTYPE_X, DTYPE_CACHE, DTYPE_CACHE_SCALE> op;
        op.Init(x, slot_mapping, cache, cache_scale, userWs, &tilingData, &pipe);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
