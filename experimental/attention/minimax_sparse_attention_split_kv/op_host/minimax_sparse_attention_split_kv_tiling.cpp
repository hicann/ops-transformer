/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * =========================================================================================
 * minimax_sparse_attention_split_kv -- Tiling Entry Point
 * =========================================================================================
 *
 * This file contains ONLY the tiling entry function and architecture dispatch.
 * All tiling logic is implemented in architecture-specific files:
 *
 *   A5 (ascend950): minimax_sparse_attention_split_kv_tiling_a5.cpp
 *     - L0C->UB direct Fixpipe, no GM S staging workspace
 *     - bf16 only, D=128 only
 *     - blockDim = min(totalTaskMax, aicNum_)
 *
 *   A2 (ascend910b): minimax_sparse_attention_split_kv_tiling_a2.cpp
 *     - L0C->GM->UB staging, requires GM S/P staging workspace
 *     - bf16+fp16, D=128/256, supports continuous KV (blockSize=0)
 *     - blockDim = aicNum_ (SyncAll requires all cores launched)
 *
 * The two implementations are completely independent -- no shared base class,
 * no virtual methods, no cross-references.  Modifying one cannot affect the other.
 * =========================================================================================
 */

#include "minimax_sparse_attention_split_kv_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"

namespace optiling {

// =================================================================================
// Entry: architecture detection -> dispatch to A5 or A2 tiling
//
// A5 (ascend950/DAV_3510 Regbase): IsRegbaseSocVersion returns true
// A2 (ascend910b): IsRegbaseSocVersion returns false
// =================================================================================

ASCENDC_EXTERN_C ge::graphStatus TilingMinimaxSparseAttentionSplitKv(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Architecture detection: A5 (Regbase) returns true, A2 returns false
    const bool isA5 = Ops::Transformer::OpTiling::IsRegbaseSocVersion(context);

    MinimaxSparseAttentionSplitKvTilingData tilingData;

    if (isA5) {
        MinimaxSaSplitKvTilingA5 tiling;
        if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
            tiling.SetTilingData(context, tilingData);
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGE(context->GetNodeName(), "A5 GetTiling failed");
    } else {
        MinimaxSaSplitKvTilingA2 tiling;
        if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
            tiling.SetTilingData(context, tilingData);
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGE(context->GetNodeName(), "A2 GetTiling failed");
    }

    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMinimaxSparseAttentionSplitKv(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MinimaxSparseAttentionSplitKv)
    .Tiling(TilingMinimaxSparseAttentionSplitKv)
    .TilingParse<MinimaxSparseAttentionSplitKvCompileInfo>(TilingPrepareForMinimaxSparseAttentionSplitKv);

} // namespace optiling
