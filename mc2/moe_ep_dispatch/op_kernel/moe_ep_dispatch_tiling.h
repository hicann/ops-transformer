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
 * \file moe_ep_dispatch_tiling.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_TILING_H
#define MOE_EP_DISPATCH_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

constexpr uint32_t MOE_EP_SEND_ENTRY_BYTES = 2U * sizeof(uint32_t); // source slot和destination slot

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

struct MoeEpDispatchInfo {
    MoeEpCommonTilingData cfg;
    uint32_t doCpuSync;
    uint32_t isCached;
    uint32_t isTopkWeights;
    uint32_t isMxQuant;
    uint32_t networkMode;
    uint32_t rankSizePerServer; // 单超节点内rank数
    uint32_t numScaleupRanks;
    uint32_t numScaleoutRanks;
    uint32_t numAivStage1;
    uint32_t numAivStage2;
    uint32_t aivNum;
    uint32_t dispatchNotifyCount;
    uint32_t scaleoutSlotAlignedBytes;   // scaleout 数据和转发元信息 slot 字节数
    uint32_t fanoutCountCoreBytes;       // 单个 fanout 生产核的计数区字节数
    uint64_t sendEntryTokenRangeBytes;  // 单个 token 范围的发送记录字节数
    uint64_t fanoutSendEntryCoreBytes;  // 单个 fanout 生产核的发送记录字节数
    uint64_t hostPinnedCounterAddr;
    uint64_t routeWorkspaceOffset;
    uint64_t scaleoutSendEntryOffset;
    uint64_t scaleupSendEntryOffset;
    uint64_t fanoutSendEntryOffset;
    uint64_t fanoutCountOffset;
    uint64_t cntWinStateOffset;
    uint64_t slotWinStateOffset;
    uint64_t winDataOffset;
    uint64_t scaleoutRecvDataOffset;
    uint64_t scaleupFinalRecvDataOffset;
    uint64_t scaleoutRecvStatusOffset;
    uint64_t combineDataOffset;
    uint64_t payloadStashWinOffset;
    uint64_t totalWinSizeEp;
    uint64_t totalUbSize;
};

struct MoeEpDispatchTilingData {
    MoeEpDispatchInfo moeEpDispatchInfo;
};

#endif // MOE_EP_DISPATCH_TILING_H
