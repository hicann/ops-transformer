/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_AIV_COMM_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_AIV_COMM_H

#include <cstdint>

namespace AlltoAllvGroupedMatMulAiv {

constexpr uint32_t kMaxRankSize = 128U;
constexpr uint32_t kMaxA2RankSize = 8U;
constexpr uint64_t kWindowAlignment = 512U;
constexpr uint64_t kFlagSlotBytes = 32U;
constexpr uint64_t kControlFlagStride = kFlagSlotBytes;
constexpr uint64_t kEpochOffset = 0U;
constexpr uint64_t kPublishSlotsOffset = kControlFlagStride;
constexpr uint64_t kInputElementBytes = 2U;

struct ExpertMeta {
    uint64_t recvTokenBase = 0U;
    uint32_t tokenCount = 0U;
    uint32_t reserved = 0U;
};

struct WorkspaceLayout {
    uint64_t payloadBytes = 0U;
    uint64_t totalBytes = 0U;
};

struct A2avWindowLayout {
    uint64_t inputBytes = 0U;
    uint64_t payloadBytes = 0U;
    uint64_t countsOffset = 0U;
    uint64_t countBytes = 0U;
    uint64_t controlOffset = 0U;
    uint64_t controlBytes = 0U;
    uint64_t totalBytes = 0U;
    uint64_t requiredBytes = 0U;
};

struct RuntimeControlLayout {
    uint64_t epochOffset = 0U;
    uint64_t publishSlotsOffset = 0U;
    uint64_t releaseSlotsOffset = 0U;
    // 对端窗口中仅保存 epoch、发布和释放状态。此边界必须与每卡专家数无关，确保连续运行不同形状时复用同一跨 rank
    // 同步地址。
    uint64_t peerControlBytes = 0U;
};

struct PeerContextMetadata {
    uint32_t rankId = 0U;
    uint32_t rankSize = 0U;
    uint64_t windowBytes = 0U;
};

#if defined(__CCE_AICORE__)
#define A2AVGMM_HOST_DEVICE __aicore__ inline
#else
#define A2AVGMM_HOST_DEVICE inline
#endif

A2AVGMM_HOST_DEVICE bool IsSupportedPeerRankSize(uint32_t rankSize, bool isA3)
{
    const bool supportedByA2 = rankSize == 2U || rankSize == 4U || rankSize == kMaxA2RankSize;
    const bool supportedByA3 = rankSize == 16U || rankSize == 32U || rankSize == 64U || rankSize == kMaxRankSize;
    return supportedByA2 || (isA3 && supportedByA3);
}

A2AVGMM_HOST_DEVICE bool NormalizePeerContextMetadata(uint32_t rankId, uint32_t rankSize, uint64_t windowBytes,
                                                      PeerContextMetadata &metadata)
{
    if (rankSize == 0U || rankSize > kMaxRankSize || rankId >= rankSize || windowBytes == 0U) {
        return false;
    }
    metadata.rankId = rankId;
    metadata.rankSize = rankSize;
    metadata.windowBytes = windowBytes;
    return true;
}

A2AVGMM_HOST_DEVICE bool SafeAddU64(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t maxValue = ~static_cast<uint64_t>(0U);
    if (lhs > maxValue - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

A2AVGMM_HOST_DEVICE bool SafeMulU64(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t maxValue = ~static_cast<uint64_t>(0U);
    if (lhs != 0U && rhs > maxValue / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

A2AVGMM_HOST_DEVICE bool BuildRuntimeControlLayout(uint32_t rankSize, uint32_t expertPerRank,
                                                   RuntimeControlLayout &layout)
{
    if (rankSize == 0U || rankSize > kMaxRankSize || expertPerRank == 0U) {
        return false;
    }
    uint64_t rankSlotsBytes = 0U;
    layout = {};
    layout.epochOffset = kEpochOffset;
    layout.publishSlotsOffset = kPublishSlotsOffset;
    if (!SafeMulU64(rankSize, kFlagSlotBytes, rankSlotsBytes) ||
        !SafeAddU64(layout.publishSlotsOffset, rankSlotsBytes, layout.releaseSlotsOffset) ||
        !SafeAddU64(layout.releaseSlotsOffset, rankSlotsBytes, layout.peerControlBytes)) {
        layout = {};
        return false;
    }
    return true;
}

A2AVGMM_HOST_DEVICE uint64_t MulU32ToU64(uint32_t lhs, uint32_t rhs)
{
    const uint32_t lhsLow = lhs & 0xffffU;
    const uint32_t lhsHigh = lhs >> 16U;
    const uint32_t rhsLow = rhs & 0xffffU;
    const uint32_t rhsHigh = rhs >> 16U;
    const uint64_t low = static_cast<uint64_t>(lhsLow * rhsLow);
    const uint64_t middle = static_cast<uint64_t>(lhsLow * rhsHigh) + static_cast<uint64_t>(lhsHigh * rhsLow);
    const uint64_t high = static_cast<uint64_t>(lhsHigh * rhsHigh);
    return low + (middle << 16U) + (high << 32U);
}

A2AVGMM_HOST_DEVICE bool AlignUpU64(uint64_t value, uint64_t alignment, uint64_t &result)
{
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        return false;
    }
    uint64_t valueWithPadding = 0U;
    if (!SafeAddU64(value, alignment - 1U, valueWithPadding)) {
        return false;
    }
    result = valueWithPadding & ~(alignment - 1U);
    return true;
}

A2AVGMM_HOST_DEVICE uint64_t AlignDownU64(uint64_t value, uint64_t alignment)
{
    return value & ~(alignment - 1U);
}

A2AVGMM_HOST_DEVICE bool BuildWorkspaceLayout(uint64_t tokenNum, uint64_t hiddenSize, uint64_t dtypeBytes,
                                              WorkspaceLayout &layout)
{
    uint64_t elementNum = 0U;
    uint64_t rawPayloadBytes = 0U;
    uint64_t payloadBytes = 0U;
    layout = {};
    if (!SafeMulU64(tokenNum, hiddenSize, elementNum) || !SafeMulU64(elementNum, dtypeBytes, rawPayloadBytes) ||
        !AlignUpU64(rawPayloadBytes, kWindowAlignment, payloadBytes)) {
        return false;
    }
    layout.payloadBytes = payloadBytes;
    layout.totalBytes = payloadBytes;
    return true;
}

template <typename PrefixPtr>
A2AVGMM_HOST_DEVICE bool PrefixRange(PrefixPtr inclusivePrefix, uint32_t countNum, uint32_t index, uint32_t &begin,
                                     uint32_t &end)
{
    begin = 0U;
    end = 0U;
    if (inclusivePrefix == nullptr || countNum == 0U || index >= countNum) {
        return false;
    }

    const int32_t current = inclusivePrefix[index];
    const int32_t previous = index == 0U ? 0 : inclusivePrefix[index - 1U];
    if (previous < 0 || current < previous) {
        return false;
    }
    begin = static_cast<uint32_t>(previous);
    end = static_cast<uint32_t>(current);
    return true;
}

A2AVGMM_HOST_DEVICE bool BuildWindowLayout(uint64_t inputTokenNum, uint64_t hiddenSize, uint64_t countNum,
                                           const RuntimeControlLayout &control, uint64_t windowBytes,
                                           A2avWindowLayout &layout)
{
    uint64_t rawCountBytes = 0U;
    uint64_t countAreaBytes = 0U;
    uint64_t requiredBytes = 0U;
    uint64_t payloadAndCounts = 0U;
    uint64_t controlBytes = 0U;
    WorkspaceLayout workspace = {};
    layout = {};
    if (!BuildWorkspaceLayout(inputTokenNum, hiddenSize, kInputElementBytes, workspace) ||
        !SafeMulU64(countNum, sizeof(int32_t), rawCountBytes) ||
        !AlignUpU64(rawCountBytes, kWindowAlignment, countAreaBytes) ||
        !AlignUpU64(control.peerControlBytes, kWindowAlignment, controlBytes) ||
        !SafeAddU64(workspace.payloadBytes, countAreaBytes, payloadAndCounts) ||
        !SafeAddU64(payloadAndCounts, controlBytes, requiredBytes)) {
        return false;
    }
    if (control.peerControlBytes == 0U || control.peerControlBytes > controlBytes) {
        return false;
    }

    layout.inputBytes = workspace.payloadBytes;
    layout.payloadBytes = workspace.payloadBytes;
    layout.countBytes = countAreaBytes;
    layout.controlBytes = controlBytes;
    layout.totalBytes = requiredBytes;
    layout.requiredBytes = requiredBytes;
    if (windowBytes == 0U) {
        return true;
    }
    if (requiredBytes > windowBytes || controlBytes > windowBytes) {
        layout = {};
        return false;
    }
    layout.controlOffset = AlignDownU64(windowBytes - controlBytes, kWindowAlignment);
    if (layout.controlOffset < countAreaBytes) {
        layout = {};
        return false;
    }
    layout.countsOffset = layout.controlOffset - countAreaBytes;
    if (layout.payloadBytes > layout.countsOffset) {
        layout = {};
        return false;
    }
    return true;
}
#undef A2AVGMM_HOST_DEVICE

} // namespace AlltoAllvGroupedMatMulAiv

#if defined(__CCE_AICORE__)

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"

namespace AlltoAllvGroupedMatMulAiv {

class PeerWindowContext {
public:
    __aicore__ inline bool Init(uint32_t is910C, uint64_t queriedWindowBytes)
    {
        GM_ADDR contextAddress = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        if (contextAddress == nullptr) {
            return false;
        }

        PeerContextMetadata metadata = {};
        if (is910C != 0U) {
            a3Context_ = reinterpret_cast<__gm__ AscendC::HcclContextDef::HcclOpResParam *>(contextAddress);
            if (!NormalizePeerContextMetadata(a3Context_->rankId, a3Context_->rankNum,
                                              a3Context_->winSize == 0U ? queriedWindowBytes : a3Context_->winSize,
                                              metadata)) {
                return false;
            }
            isA3_ = true;
        } else {
            a2Context_ = reinterpret_cast<__gm__ AscendC::HcclCombineOpParam *>(contextAddress);
            if (!NormalizePeerContextMetadata(a2Context_->rankId, a2Context_->rankNum,
                                              a2Context_->winSize == 0U ? queriedWindowBytes : a2Context_->winSize,
                                              metadata)) {
                return false;
            }
            // 在 A2 上，CANN 也可能选择动态对端窗口表，例如 EP8 通信域且 count_num 为 128 时。EP
            // 能力由IsSupportedPeerRankSize 单独约束； 窗口地址表示仍须遵循运行时 HCCL 上下文。
            if (a2Context_->multiFlag != 0U && a2Context_->data == nullptr) {
                return false;
            }
        }
        // 运行时窗口大小为零时使用 Host 成功查询的结果。窗口变化会使公共尾部控制区地址失效。
        if (metadata.windowBytes != queriedWindowBytes || !IsSupportedPeerRankSize(metadata.rankSize, isA3_)) {
            return false;
        }

        rankId_ = metadata.rankId;
        rankSize_ = metadata.rankSize;
        windowBytes_ = metadata.windowBytes;
        for (uint32_t rank = 0U; rank < rankSize_; ++rank) {
            if (Window(rank) == nullptr) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint32_t RankId() const
    {
        return rankId_;
    }

    __aicore__ inline uint32_t RankSize() const
    {
        return rankSize_;
    }

    __aicore__ inline uint64_t WindowBytes() const
    {
        return windowBytes_;
    }

    __aicore__ inline GM_ADDR Window(uint32_t rank) const
    {
        if (rank >= rankSize_) {
            return nullptr;
        }
        if (isA3_) {
            if (rank == rankId_) {
                return reinterpret_cast<GM_ADDR>(a3Context_->localWindowsIn);
            }
            auto *relation = AscendC::GetRemoteRankAddrs(a3Context_, rank);
            return relation == nullptr ? nullptr : reinterpret_cast<GM_ADDR>(relation->windowsIn);
        }
        if (a2Context_->multiFlag == 0U) {
            return reinterpret_cast<GM_ADDR>(a2Context_->windowsIn[rank]);
        }
        return rank == rankId_ ? reinterpret_cast<GM_ADDR>(a2Context_->data[rank].localInput.addr) :
                                 reinterpret_cast<GM_ADDR>(a2Context_->data[rank].remoteInput.addr);
    }

private:
    __gm__ AscendC::HcclCombineOpParam *a2Context_ = nullptr;
    __gm__ AscendC::HcclContextDef::HcclOpResParam *a3Context_ = nullptr;
    uint32_t rankId_ = 0U;
    uint32_t rankSize_ = 0U;
    uint64_t windowBytes_ = 0U;
    bool isA3_ = false;
};

template <typename T>
__aicore__ inline void CopyPeerGmToLocalGm(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, uint64_t count,
                                           uint32_t ubMoveNum, AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer0,
                                           AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer1)
{
    if (count == 0U || ubMoveNum == 0U) {
        return;
    }

    // 只有一个数据块时无法重叠搬运，无需初始化和等待第二组事件。
    if (count <= ubMoveNum) {
        AscendC::LocalTensor<T> local = buffer0.template Get<T>();
        AscendC::DataCopyExtParams copyParams(1U, static_cast<uint32_t>(count) * sizeof(T), 0U, 0U, 0U);
        AscendC::DataCopyPadExtParams<T> padParams;
        AscendC::DataCopyPad(local, src, copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst, local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        return;
    }
    AscendC::LocalTensor<T> local0 = buffer0.template Get<T>();
    AscendC::LocalTensor<T> local1 = buffer1.template Get<T>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t pingPongId = 0U;
    for (uint64_t offset = 0U; offset < count; offset += ubMoveNum) {
        const uint32_t current = static_cast<uint32_t>((count - offset) > ubMoveNum ? ubMoveNum : (count - offset));
        const uint32_t copyBytes = current * sizeof(T);
        const AscendC::TEventID eventId = pingPongId == 0U ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<T> local = pingPongId == 0U ? local0 : local1;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        AscendC::DataCopyExtParams copyParams(1U, copyBytes, 0U, 0U, 0U);
        AscendC::DataCopyPadExtParams<T> padParams;
        AscendC::DataCopyPad(local, src[offset], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::DataCopyPad(dst[offset], local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        pingPongId ^= 1U;
    }
    // 发布就绪状态或复用事件编号前，等待两个缓冲区的搬运全部完成。
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void CopyLocalGmToWindow(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, uint64_t count,
                                           uint32_t ubMoveNum, AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer0,
                                           AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer1)
{
    CopyPeerGmToLocalGm(dst, src, count, ubMoveNum, buffer0, buffer1);
}

__aicore__ inline __gm__ int32_t *GetFlagAddress(GM_ADDR window, uint64_t controlOffset, uint64_t flagOffset)
{
    return reinterpret_cast<__gm__ int32_t *>(window + controlOffset + flagOffset);
}

__aicore__ inline void StoreFlag(__gm__ int32_t *address, AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer,
                                 int32_t value)
{
    AscendC::LocalTensor<int32_t> local = flagBuffer.Get<int32_t>();
    local.SetValue(0U, value);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(address);
    AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
    AscendC::DataCopyPad(global, local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
}

__aicore__ inline int32_t LoadFlag(__gm__ int32_t *address, AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    AscendC::LocalTensor<int32_t> local = flagBuffer.Get<int32_t>();
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(address);
    AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
    AscendC::DataCopyPadExtParams<int32_t> padParams;
    AscendC::DataCopyPad(local, global, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    return local.GetValue(0U);
}

__aicore__ inline void WaitFlagValue(__gm__ int32_t *address, AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer,
                                     int32_t value)
{
    while (LoadFlag(address, flagBuffer) != value) {
    }
}

__aicore__ inline int32_t NextInvocationEpoch(const PeerWindowContext &context, const A2avWindowLayout &layout,
                                              AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    __gm__ int32_t *epochAddress = GetFlagAddress(context.Window(context.RankId()), layout.controlOffset, kEpochOffset);
    const int32_t current = LoadFlag(epochAddress, flagBuffer);
    const int32_t next = (current <= 0 || current == 0x7fffffff) ? 1 : current + 1;
    StoreFlag(epochAddress, flagBuffer, next);
    return next;
}

__aicore__ inline void PublishPeerPhase(const PeerWindowContext &context, const A2avWindowLayout &layout,
                                        uint64_t phaseOffset, int32_t epoch, uint32_t workerIdx, uint32_t workerNum,
                                        uint32_t subBlockIdx, AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    if (subBlockIdx != 0U || workerNum == 0U || workerIdx >= workerNum) {
        return;
    }
    for (uint32_t rank = workerIdx; rank < context.RankSize(); rank += workerNum) {
        __gm__ int32_t *slot =
            GetFlagAddress(context.Window(rank), layout.controlOffset,
                           phaseOffset + static_cast<uint64_t>(context.RankId()) * kControlFlagStride);
        StoreFlag(slot, flagBuffer, epoch);
    }
}

__aicore__ inline void WaitPeerPhase(const PeerWindowContext &context, const A2avWindowLayout &layout,
                                     uint64_t phaseOffset, int32_t epoch,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    for (uint32_t sourceRank = 0U; sourceRank < context.RankSize(); ++sourceRank) {
        __gm__ int32_t *slot = GetFlagAddress(context.Window(context.RankId()), layout.controlOffset,
                                              phaseOffset + static_cast<uint64_t>(sourceRank) * kControlFlagStride);
        WaitFlagValue(slot, flagBuffer, epoch);
    }
}

} // namespace AlltoAllvGroupedMatMulAiv

#endif // defined(__CCE_AICORE__)

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_AIV_COMM_H
