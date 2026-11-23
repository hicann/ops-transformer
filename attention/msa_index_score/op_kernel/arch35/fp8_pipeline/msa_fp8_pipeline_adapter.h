/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// MsaIndexScore ABI adapter for the native FP8 pipeline.
#pragma once
#include "msa_fp8_pipeline.h"

namespace MsaIndexScoreNs {
template <class ElementQ, bool FUSED_SELECT = false, bool BALANCED = false, bool BYPASS = false>
__aicore__ inline bool RunNativeFp8Pipeline(GM_ADDR query, GM_ADDR key, GM_ADDR table, GM_ADDR qLengths,
                                            GM_ADDR kLengths, GM_ADDR score, GM_ADDR workspace, GM_ADDR indices,
                                            GM_ADDR values, const MsaIndexScoreTilingData* tiling)
{
    AscendC::GlobalTensor<int32_t> qPrefix;
    qPrefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qLengths));
    if (tiling->strideKvToken != 128U || qPrefix.GetValue(0) != 0 ||
        qPrefix.GetValue(tiling->batch) != static_cast<int32_t>(tiling->totalQ)) {
        return false;
    }
    MsaFp8PipelineArgs a{};
    a.q = reinterpret_cast<uint64_t>(query);
    a.k = reinterpret_cast<uint64_t>(key);
    a.table = reinterpret_cast<uint64_t>(table);
    a.out = !FUSED_SELECT ? reinterpret_cast<uint64_t>(score) :
                            reinterpret_cast<uint64_t>(workspace) +
                                static_cast<uint64_t>(tiling->scoreWorkspaceElems) * sizeof(float);
    a.q_prefix = reinterpret_cast<uint64_t>(qLengths);
    a.k_lengths = reinterpret_cast<uint64_t>(kLengths);
    a.tokens = tiling->totalQ;
    a.batches = tiling->batch;
    a.pages = tiling->numPages;
    a.table_columns = tiling->tableWidth;
    a.scores = !FUSED_SELECT ? tiling->strideOutToken : tiling->scoreBlockStride;
    a.q_stride = tiling->strideQt;
    a.k_stride = tiling->strideKvBlock;
    a.table_stride = tiling->tableWidth;
    a.out_stride = a.scores;
    a.query_tile = 8;
    a.summary_tile = 256;
    a.causal = tiling->sparseMode == 3U;
    MsaNativeFp8Pipeline::Pipeline<ElementQ, BALANCED, BYPASS>(a);
    if constexpr (FUSED_SELECT) {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();
#if !defined(__DAV_C310_CUBE__) && !defined(__DAV_310R6_CUBE__)
        MsaBlock16Select::FusedSelect::VisibleMeta meta;
        meta.actualSeqQ = qLengths;
        meta.actualSeqK = kLengths;
        meta.batch = tiling->batch;
        meta.sparseMode = tiling->sparseMode;
        meta.initBlocks = tiling->initBlocks;
        meta.localBlocks = tiling->localBlocks;
        meta.compressFactor = tiling->compressFactor;
        meta.keyLayout = tiling->keyLayout;
        meta.qMaxSeq = tiling->qMaxSeq;
        meta.kMaxSeq = tiling->kMaxSeq;
        meta.maxBlocksPerBatch = tiling->maxBlocksPerBatch;
        meta.tableWidth = tiling->tableWidth;
        meta.pageSize = tiling->pageSize;
        MsaBlock16Select::FusedSelect::Run(reinterpret_cast<GM_ADDR>(a.out), indices,
                                           tiling->returnChoice == MSA_RETURN_INDICES_VALUES ? values : nullptr,
                                           tiling->totalQ, static_cast<uint32_t>(a.out_stride), AscendC::GetBlockNum(),
                                           tiling->returnChoice == MSA_RETURN_INDICES_VALUES, meta);
#endif
    }
    return true;
}
} // namespace MsaIndexScoreNs
