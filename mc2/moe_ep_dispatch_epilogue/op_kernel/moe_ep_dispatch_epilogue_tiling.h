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
    // 占位，勿删：这个字段曾经死在 metadataRankOffsetsOffset 前面，删掉会把后者前移 8 字节。
    // tiling 结构体是 host tiling 与 kernel 共用的裸内存 ABI，偏移一变，两侧只要不是同一次构建，
    // kernel 就会把 metadataRankOffsetsOffset 读成 0，offsets 被写到 recvSrcMetadata + 0，
    // 随后又被 metadata 行覆盖，表现为 recv_rank_offsets 恒为 torch.empty 的 [0, 0, 0]。
    // 保留占位即可让新旧 host/kernel 混用时字段偏移不变。
    uint64_t rankExpertHitCountOffsetReserved = 0;
    uint64_t metadataRankOffsetsOffset = 0; // byte offset of the aligned tail in packed metadata
};

struct MoeEpDispatchEpilogueTilingData {
    MoeEpDispatchEpilogueInfo moeEpDispatchEpilogueInfo;
};

#endif // MOE_EP_DISPATCH_EPILOGUE_TILING_H
