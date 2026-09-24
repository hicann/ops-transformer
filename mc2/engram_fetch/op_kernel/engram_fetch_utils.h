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
 * \file engram_fetch_utils.h
 * \brief engram_fetch算子公共头文件
 */

#ifndef ENGRAM_FETCH_UTILS_H
#define ENGRAM_FETCH_UTILS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Mc2Kernel {
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t TILE_BYTES = 32U * 1024U;
constexpr uint32_t TILE_BYTES_INFER_MAX = 24U * 1024U;
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t ENGRAM_BATCH_CAPACITY = 16U;
constexpr uint32_t ENGRAM_WQE_BYTES = 64U;
constexpr uint32_t ENGRAM_BATCH_BUFFER_BYTES = ENGRAM_BATCH_CAPACITY * ENGRAM_WQE_BYTES;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t RELAY_BUFFER_NUM = 2U;
constexpr uint32_t RELAY_BUFFER_NUM_INFER = 6U;

constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t WIN_REGION_COUNT = 6U;
constexpr uint32_t NUM_SLOTS = 4U;
constexpr uint32_t INDICES_RATIO = 50U;
constexpr uint32_t UB_RESERVED_SIZE = 8U * 1024U;
constexpr uint32_t MAX_CHANNELS_PER_RANK = 3U;
constexpr uint32_t HCOMM_SQ_MAX_PENDING = 32767U;

struct EngramCommContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE];
    uint32_t channelsPerRank;
};

struct CoreAssignment {
    uint32_t assignedRank;   // 本核负责的卡
    uint32_t idxInRankGroup; // 远端核=通道号；本地核=本地token分片序号
    uint32_t rankGroupSize;  // 远端核=该卡通道数；本地核=本地核总数
};

__aicore__ inline CoreAssignment GetCoreAssignment(uint32_t totalBlocks, uint32_t aivId, uint32_t numRanks,
                                                   uint32_t rankId, uint32_t channelsPerRank)
{
    if (numRanks <= 1U) {
        return CoreAssignment{rankId, aivId, totalBlocks};
    }
    // 每张对端卡分到的通道数(=负责该卡的核数)，三者取小：
    // (totalBlocks-1)/(numRanks-1)：核预算，预留1核干本地拷贝后均摊
    // MAX_CHANNELS_PER_RANK：限制单卡通道上限
    // channelsPerRank：handle资源摊到每卡的实际通道数
    uint32_t maxChannel = (channelsPerRank < MAX_CHANNELS_PER_RANK) ? channelsPerRank : MAX_CHANNELS_PER_RANK;
    uint32_t usedChannelPerRank = (totalBlocks - 1U) / (numRanks - 1U);
    usedChannelPerRank = (usedChannelPerRank < maxChannel) ? usedChannelPerRank : maxChannel;
    uint32_t urmaCores = (numRanks - 1U) * usedChannelPerRank;
    if (aivId < urmaCores) {
        uint32_t rankIdx = aivId / usedChannelPerRank;
        uint32_t assignedRank = (rankIdx < rankId) ? rankIdx : (rankIdx + 1U); // 跳过本卡
        return CoreAssignment{assignedRank, aivId - rankIdx * usedChannelPerRank, usedChannelPerRank};
    }
    // 本地核：全部服务本卡，按(aivId-urmaCores)分片本地token
    return CoreAssignment{rankId, aivId - urmaCores, totalBlocks - urmaCores};
}

} // namespace Mc2Kernel

#endif
