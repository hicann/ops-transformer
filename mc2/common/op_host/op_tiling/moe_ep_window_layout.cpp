/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_ep_window_layout.h"

#include <cstring>

#include "mc2_log.h"

namespace Mc2Tiling {

uint64_t AlignMoeEpUb(uint64_t value)
{
    return (value + MOE_EP_UB_ALIGN - 1UL) / MOE_EP_UB_ALIGN * MOE_EP_UB_ALIGN;
}

uint64_t AlignMoeEpWin(uint64_t value)
{
    return (value + MOE_EP_WIN_ALIGN - 1UL) / MOE_EP_WIN_ALIGN * MOE_EP_WIN_ALIGN;
}

ge::graphStatus ResolveMoeEpTopology(uint32_t epWorldSize, int64_t requestedNetworkMode, int64_t rankNumPerServer,
                                     MoeEpTopology &topology)
{
    if ((requestedNetworkMode != MOE_EP_NETWORK_DIRECT && requestedNetworkMode != MOE_EP_NETWORK_HYBRID) ||
        rankNumPerServer <= 0 || epWorldSize % static_cast<uint64_t>(rankNumPerServer) != 0UL) {
        return ge::GRAPH_FAILED;
    }
    topology.rankNumPerServer = static_cast<uint32_t>(rankNumPerServer);
    topology.serverNum = epWorldSize / topology.rankNumPerServer;
    topology.networkMode = requestedNetworkMode == MOE_EP_NETWORK_HYBRID && topology.serverNum > 1U ?
                               MOE_EP_NETWORK_HYBRID :
                               MOE_EP_NETWORK_DIRECT;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcMoeEpWindowLayout(const MoeEpWindowLayoutParams &params, MoeEpWindowLayout &layout)
{
    std::memset(&layout, 0, sizeof(layout));
    layout.dispatchNotifyCount = static_cast<uint32_t>(
        (static_cast<uint64_t>(params.nmt) + MOE_EP_NOTIFY_TOKEN_LIMIT - 1UL) / MOE_EP_NOTIFY_TOKEN_LIMIT);

    layout.metadataOffset = 0UL;
    layout.metadataSize = MOE_EP_DUMP_METADATA_BYTES;
    layout.perCoreDiagOffset = layout.metadataSize;
    layout.perCoreDiagSize = MOE_EP_PER_CORE_DIAG_BYTES;

    const uint64_t epWorldSize = params.epWorldSize;
    const uint64_t nmt = params.nmt;
    const uint64_t topK = params.topK;
    const uint64_t serverNum = params.serverNum;
    const uint64_t countNotifySize = epWorldSize * MOE_EP_WIN_ALIGN;
    const uint64_t expertCountSize = epWorldSize * AlignMoeEpWin(params.localExpertNum * sizeof(int32_t));

    layout.cntWinStateOffset = MOE_EP_FIXED_PREFIX_BYTES;
    layout.cntWinStateSize = countNotifySize + expertCountSize;
    layout.slotWinStateOffset = layout.cntWinStateOffset + layout.cntWinStateSize;
    layout.dispatchSlotStateSize = layout.dispatchNotifyCount * epWorldSize * MOE_EP_WIN_ALIGN;
    layout.combineStateWinOffset = layout.slotWinStateOffset + layout.dispatchSlotStateSize;
    layout.combineStateWinSize = nmt * topK * MOE_EP_WIN_ALIGN + epWorldSize * MOE_EP_WIN_ALIGN;

    const uint64_t dataBase = layout.combineStateWinOffset + layout.combineStateWinSize;
    const uint64_t hiddenAlign = AlignMoeEpUb(params.hidden * MOE_EP_MAX_OUT_DTYPE_SIZE);
    const uint64_t topKAlign = AlignMoeEpUb(topK * MOE_EP_METADATA_DTYPE_SIZE);
    layout.dispatchReservedPerSlotBytes = AlignMoeEpWin(hiddenAlign + 2UL * topKAlign + MOE_EP_UB_ALIGN);
    layout.scaleoutReservedPerSlotBytes = AlignMoeEpWin(layout.dispatchReservedPerSlotBytes + topK * sizeof(int32_t));
    layout.combineReservedPerSlotBytes = AlignMoeEpWin(hiddenAlign + MOE_EP_UB_ALIGN);
    layout.dispatchRecvDataSize = epWorldSize * nmt * layout.dispatchReservedPerSlotBytes;
    layout.combineDataSize = nmt * topK * layout.combineReservedPerSlotBytes;

    if (params.networkMode == MOE_EP_NETWORK_HYBRID) {
        layout.scaleoutRecvDataOffset = dataBase;
        layout.scaleoutRecvDataSize = serverNum * nmt * layout.scaleoutReservedPerSlotBytes;
        layout.winDataOffset = layout.scaleoutRecvDataOffset + layout.scaleoutRecvDataSize;
        layout.scaleoutRecvStatusOffset = layout.winDataOffset + layout.dispatchRecvDataSize;
        layout.scaleoutRecvStatusSize = serverNum * nmt * MOE_EP_WIN_ALIGN;
        layout.combineDataWinOffset = layout.scaleoutRecvStatusOffset + layout.scaleoutRecvStatusSize;
        layout.payloadStashWinOffset = layout.combineDataWinOffset + layout.combineDataSize;
        layout.payloadStashWinSize = nmt * layout.scaleoutReservedPerSlotBytes;
    } else {
        layout.winDataOffset = dataBase;
        layout.combineDataWinOffset = layout.winDataOffset + layout.dispatchRecvDataSize;
        layout.payloadStashWinOffset = layout.combineDataWinOffset + layout.combineDataSize;
        layout.payloadStashWinSize = layout.dispatchRecvDataSize;
    }
    layout.requiredBytes = layout.payloadStashWinOffset + layout.payloadStashWinSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMoeEpWindowCapacity(uint64_t requiredBytes, uint64_t cclBufferSize, const char *nodeName)
{
    if (requiredBytes > cclBufferSize) {
        OP_LOGE(nodeName, "ccl_buffer_size is not enough, need %lu bytes, but got %lu bytes.", requiredBytes,
                cclBufferSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

MoeEpDumpMetadata BuildMoeEpDumpMetadata(const MoeEpWindowLayoutParams &params, const MoeEpWindowLayout &layout,
                                         uint32_t aivNum)
{
    MoeEpDumpMetadata metadata{};
    metadata.layoutVersion = MOE_EP_DUMP_LAYOUT_VERSION;
    metadata.nmt = params.nmt;
    metadata.topK = params.topK;
    metadata.hidden = params.hidden;
    metadata.epWorldSize = params.epWorldSize;
    metadata.localExpertNum = params.localExpertNum;
    metadata.aivNum = aivNum;
    metadata.networkMode = params.networkMode;
    metadata.serverNum = params.serverNum;
    metadata.rankNumPerServer = params.rankNumPerServer;

    const uint64_t countNotifySize = static_cast<uint64_t>(params.epWorldSize) * MOE_EP_WIN_ALIGN;
    const uint64_t combineTokenStateSize = static_cast<uint64_t>(params.nmt) * params.topK * MOE_EP_WIN_ALIGN;
    metadata.regions[MOE_EP_DUMP_REGION_PER_CORE_DIAG] = {layout.perCoreDiagOffset, layout.perCoreDiagSize};
    metadata.regions[MOE_EP_DUMP_REGION_COUNT_NOTIFY] = {layout.cntWinStateOffset, countNotifySize};
    metadata.regions[MOE_EP_DUMP_REGION_EXPERT_COUNT] = {layout.cntWinStateOffset + countNotifySize,
                                                         layout.cntWinStateSize - countNotifySize};
    metadata.regions[MOE_EP_DUMP_REGION_DISPATCH_SLOT_STATE] = {layout.slotWinStateOffset,
                                                                layout.dispatchSlotStateSize};
    metadata.regions[MOE_EP_DUMP_REGION_COMBINE_TOKEN_STATE] = {layout.combineStateWinOffset, combineTokenStateSize};
    metadata.regions[MOE_EP_DUMP_REGION_HYBRID_SCALEOUT_STATUS] = {layout.scaleoutRecvStatusOffset,
                                                                   layout.scaleoutRecvStatusSize};
    return metadata;
}

} // namespace Mc2Tiling
