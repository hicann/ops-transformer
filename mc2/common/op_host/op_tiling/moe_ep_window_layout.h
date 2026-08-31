/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_WINDOW_LAYOUT_H
#define MOE_EP_WINDOW_LAYOUT_H

#include <cstdint>

#include "graph/ge_error_codes.h"
#include "../../op_kernel/moe_ep_exception_dump_defs.h"

namespace Mc2Tiling {

constexpr uint64_t MOE_EP_UB_ALIGN = 32UL;
constexpr uint64_t MOE_EP_WIN_ALIGN = 512UL;
constexpr uint64_t MOE_EP_NOTIFY_TOKEN_LIMIT = 15000UL;
constexpr uint64_t MOE_EP_MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t MOE_EP_METADATA_DTYPE_SIZE = 4UL;
constexpr uint32_t MOE_EP_NETWORK_DIRECT = 0U;
constexpr uint32_t MOE_EP_NETWORK_HYBRID = 1U;

struct MoeEpWindowLayoutParams {
    uint32_t epWorldSize;
    uint32_t localExpertNum;
    uint32_t nmt;
    uint32_t topK;
    uint32_t hidden;
    uint32_t networkMode;
    uint32_t rankNumPerServer;
    uint32_t serverNum;
};

struct MoeEpTopology {
    uint32_t networkMode;
    uint32_t rankNumPerServer;
    uint32_t serverNum;
};

struct MoeEpWindowLayout {
    uint32_t dispatchNotifyCount;

    uint64_t metadataOffset;
    uint64_t metadataSize;
    uint64_t perCoreDiagOffset;
    uint64_t perCoreDiagSize;

    uint64_t cntWinStateOffset;
    uint64_t cntWinStateSize;
    uint64_t slotWinStateOffset;
    uint64_t dispatchSlotStateSize;
    uint64_t combineStateWinOffset;
    uint64_t combineStateWinSize;

    uint64_t scaleoutRecvDataOffset;
    uint64_t scaleoutRecvDataSize;
    uint64_t winDataOffset;
    uint64_t dispatchRecvDataSize;
    uint64_t scaleoutRecvStatusOffset;
    uint64_t scaleoutRecvStatusSize;
    uint64_t combineDataWinOffset;
    uint64_t combineDataSize;
    uint64_t payloadStashWinOffset;
    uint64_t payloadStashWinSize;

    uint64_t dispatchReservedPerSlotBytes;
    uint64_t scaleoutReservedPerSlotBytes;
    uint64_t combineReservedPerSlotBytes;
    uint64_t requiredBytes;
};

uint64_t AlignMoeEpUb(uint64_t value);
uint64_t AlignMoeEpWin(uint64_t value);

ge::graphStatus ResolveMoeEpTopology(uint32_t epWorldSize, int64_t requestedNetworkMode, int64_t rankNumPerServer,
                                     MoeEpTopology &topology);

ge::graphStatus CalcMoeEpWindowLayout(const MoeEpWindowLayoutParams &params, MoeEpWindowLayout &layout);

ge::graphStatus CheckMoeEpWindowCapacity(uint64_t requiredBytes, uint64_t cclBufferSize, const char *nodeName);

MoeEpDumpMetadata BuildMoeEpDumpMetadata(const MoeEpWindowLayoutParams &params, const MoeEpWindowLayout &layout,
                                         uint32_t aivNum);

} // namespace Mc2Tiling

#endif // MOE_EP_WINDOW_LAYOUT_H
