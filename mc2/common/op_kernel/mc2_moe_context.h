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
 * \file mc2_moe_context.h
 * \brief
 */
#ifndef MC2_MOE_CONTEXT_H
#define MC2_MOE_CONTEXT_H
#include <cstdint>

#define MC2_BARRIER_HCOMM_MIN_VERSION 90200000

namespace Mc2Aclnn {
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

constexpr uint32_t ROCE_CHANNEL_PER_SERVER = 2;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;                    // host kfc方案中，需要传递通信API所需的地址
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE]; // 地址含rank偏移；RoCE路径只填本server内的UB_MEM窗。
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE];  // 支持ROCE或者URMA
    uint32_t channelsPerRank;
    bool enableGroupSend;
    uint64_t roceHcclBuffer_[HCCL_MAX_RANK_SIZE];
    uint64_t roceBufferSize; // 本地与各对端注册窗容量一致。
    uint64_t hostStatusTableAddr;
    uint64_t completedRounds;
    uint64_t payloadBufferAddr;
    uint64_t payloadBufferSize;
    uint64_t payloadSlotStride;
    uint32_t payloadSlotNum;
};

constexpr uint32_t MC2_CONTEXT_ELEMS = sizeof(Mc2MoeContext) / sizeof(int32_t);
static_assert(MC2_CONTEXT_ELEMS % 2U == 0, "Legacy context length must be even");

struct MoeCommContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t epHcclBuffer[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE]; // 支持ROCE或者URMA
    uint32_t channelsPerRank;
};
} // namespace Mc2Aclnn

#if defined(__CCE_AICORE__)
#define MC2_BARRIER_FUNC __aicore__
#else
#define MC2_BARRIER_FUNC
#endif

namespace Mc2Barrier {
constexpr uint32_t MAGIC = 0x42415236U;
constexpr uint32_t VERSION = 6U;
constexpr uint32_t MIN_RANKS = 2U;
constexpr uint32_t MAX_RANKS = 204800U;
constexpr uint32_t BLOCK_BYTES = 32U;
constexpr uint32_t HCOMM_BYTES = 512U;
constexpr uint32_t WAIT_BATCH_SLOTS = 256U;
constexpr uint32_t NO_RANK = 0xFFFFFFFFU;
constexpr uint32_t SLOT_BYTES = 512U;
constexpr uint32_t CHANNEL_BATCH_SIZE = 512U;

MC2_BARRIER_FUNC constexpr uint32_t ElasticMaskBytes(uint32_t worldSize, uint32_t workers)
{
    const uint32_t peers = (worldSize + workers) / workers + 1U;
    return ((peers + BLOCK_BYTES * 8U - 1U) / (BLOCK_BYTES * 8U)) * BLOCK_BYTES;
}

MC2_BARRIER_FUNC constexpr uint32_t AutoUbBytes(uint32_t worldSize, uint32_t workers, bool elastic)
{
    return 2U * BLOCK_BYTES + 2U * HCOMM_BYTES + WAIT_BATCH_SLOTS * BLOCK_BYTES +
           (elastic ? ElasticMaskBytes(worldSize, workers) : 0U);
}

enum class Protocol : uint32_t {
    UBMEM,
    URMA,
    ROCE
};

MC2_BARRIER_FUNC constexpr uint32_t ChannelBatchRounds(uint32_t count)
{
    uint32_t rounds = 1U;
    while (rounds * CHANNEL_BATCH_SIZE < count) {
        rounds *= 2U;
    }
    return rounds;
}

MC2_BARRIER_FUNC constexpr uint32_t ChannelBatchBegin(uint32_t selfSlot, uint32_t round)
{
    return ((selfSlot / CHANNEL_BATCH_SIZE) ^ round) * CHANNEL_BATCH_SIZE;
}

struct Peer {
    uint64_t window = 0;
    uint64_t handle = 0;
    uint32_t rank = NO_RANK;
    Protocol protocol = Protocol::UBMEM;
};

// 后接localSize + railSize个peer，每组均含本rank。
struct Context {
    uint32_t magic = MAGIC;
    uint32_t version = VERSION;
    uint32_t worldSize = 0;
    uint32_t rank = 0;
    uint32_t localSize = 0;
    uint32_t railSize = 0;
    uint32_t reserved = 0;
    uint32_t localSlot = 0;
    uint32_t railSlot = 0;
    uint32_t protocolMask = 0;
    uint64_t window = 0;
    uint64_t windowBytes = 0;
};
static_assert(sizeof(Context) == 56 && sizeof(Peer) == 24, "Barrier context ABI changed");
static_assert(sizeof(Context) % sizeof(int32_t) == 0, "Barrier context size must be int32 aligned");

MC2_BARRIER_FUNC constexpr uint64_t ContextBytes(uint32_t localSize, uint32_t railSize)
{
    return sizeof(Context) + (static_cast<uint64_t>(localSize) + railSize) * sizeof(Peer) + sizeof(int32_t);
}

MC2_BARRIER_FUNC constexpr int64_t ContextElems(uint32_t localSize, uint32_t railSize)
{
    return ContextBytes(localSize, railSize) / sizeof(int32_t);
}

MC2_BARRIER_FUNC constexpr uint64_t BankBytes(uint32_t localSize, uint32_t railSize)
{
    return (static_cast<uint64_t>(localSize) + railSize) * SLOT_BYTES;
}

MC2_BARRIER_FUNC constexpr uint64_t PhaseOffset(uint32_t phase, uint32_t localSize)
{
    return phase == 0 ? 0 : static_cast<uint64_t>(localSize) * SLOT_BYTES;
}

MC2_BARRIER_FUNC constexpr uint64_t SourceOffset(uint32_t localSize, uint32_t railSize)
{
    return 2U * BankBytes(localSize, railSize);
}

MC2_BARRIER_FUNC constexpr uint64_t SequenceOffset(uint32_t localSize, uint32_t railSize)
{
    return SourceOffset(localSize, railSize) + SLOT_BYTES;
}

MC2_BARRIER_FUNC constexpr uint64_t WindowBytes(uint32_t localSize, uint32_t railSize)
{
    return SequenceOffset(localSize, railSize) + SLOT_BYTES;
}

#ifndef __CCE_AICORE__
inline bool IsValidContextSize(int64_t elements, uint32_t worldSize)
{
    for (uint32_t localSize = 1; localSize * localSize <= worldSize; ++localSize) {
        if (worldSize % localSize == 0 && elements == ContextElems(localSize, worldSize / localSize)) {
            return true;
        }
    }
    return false;
}

#endif
} // namespace Mc2Barrier
#undef MC2_BARRIER_FUNC

#endif // MC2_MOE_CONTEXT_H
