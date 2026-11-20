/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_DISPATCH_EPILOGUE_TILING_H
#define MOE_EP_DISPATCH_EPILOGUE_TILING_H

#include "../../common/op_kernel/moe_ep_exception_dump_defs.h"
#include <cstdint>

struct MoeEpCommonTilingData {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t numExperts;
    uint32_t numLocalExperts;
    uint32_t numTokens;
    uint32_t hidden;
    uint32_t topK;
    uint32_t numMaxTokensPerRank;
    uint32_t scalesBytes;
    uint32_t perSlotBytes;
    uint32_t expertAlignment;
};

struct MoeEpDispatchEpilogueRouteLayout {
    uint32_t token;
    uint32_t rank;
    uint32_t ends;
    uint32_t dst;
    uint32_t recvIdx;
    uint32_t topk;
    uint32_t flatRouteIndex;
    uint32_t ordinal;
    uint32_t groups;
    uint32_t metadata;
    uint32_t weights;
    uint32_t size;
};

struct MoeEpDispatchEpilogueInfo {
    MoeEpCommonTilingData cfg;
    MoeEpDumpMetadata dumpMetadata;
    uint32_t aivNum = 0;
    uint64_t totalUbSize = 0;
    uint32_t dispatchNotifyCount = 1; // slot notify count per rank
    uint64_t winDataOffset = 0;       // Win Data Offset
    uint64_t slotWinStateOffset = 0;  // slot state offset
    uint32_t cached = 0;              // 0 = non-cached path, 1 = cached path
    uint32_t isMxQuant = 0;           // 0 = float scales, 1 = fp8_e8m0 scales (MX quant)
    uint32_t networkMode = 0; // 0 = direct(紧拼槽布局 meta@tokenSize), 1 = hybrid(对齐槽布局 meta@hAlign)
    // Reserved to preserve the existing tiling data field offsets.
    uint64_t rankExpertHitCountOffsetReserved = 0;
    uint64_t metadataRankOffsetsOffset = 0; // byte offset of the aligned tail in packed metadata
    uint64_t localRecvIndexOffset = 0;      // byte offset of [num_tokens, top_k] local recv_x reverse index
    uint64_t slotEndsOffset = 0;            // compact slot list ends, capacity entries
    uint64_t slotRecvXIdsOffset = 0;        // recv-X rows grouped by source slot, capacity entries
    uint64_t totalWinSizeEp = 0;            // communication window size in bytes for OOM diagnostics

    // Append new fields here to preserve prefix offsets. Host and kernel must use the same layout.
    MoeEpDispatchEpilogueRouteLayout routeLayout; // offsets in int32 elements within the route frame
    uint32_t tokenBatch = 1U;                     // tokens per copy batch, planned from the UB budget
    uint32_t routeFrameSlots = 0U;                // source slots per route frame in the non-cached path
    uint64_t slotTopkIdsOffset = 0;               // matching top-k positions, separate compact column
};

struct MoeEpDispatchEpilogueTilingData {
    MoeEpDispatchEpilogueInfo moeEpDispatchEpilogueInfo;
};

#endif // MOE_EP_DISPATCH_EPILOGUE_TILING_H
